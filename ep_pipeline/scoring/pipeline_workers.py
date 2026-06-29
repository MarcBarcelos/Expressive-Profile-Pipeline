"""
Standalone worker functions for parallel scoring pipelines.

Each function is fully self-contained. It reads records from a shared parquet
file on disk, loads its own models, and writes results to its own checkpoint.
No shared state with the parent process or other workers.

Designed to be called via multiprocessing.Process.

Usage
-----
    from multiprocessing import Process
    from ep_pipeline.scoring.pipeline_workers import (
        worker_lingaff_synpun, worker_td, worker_sem)

    p_a = Process(target=worker_lingaff_synpun,
                  args=(records_fp, lex_dir, lingaff_fp, synpun_fp, cfg))
    p_b = Process(target=worker_td,
                  args=(records_fp, td_fp, cfg))
    p_c = Process(target=worker_sem,
                  args=(records_fp, e5_model_path, sem_fp, cfg))

    for p in (p_a, p_b, p_c): p.start()
    for p in (p_a, p_b, p_c): p.join()
"""
import traceback
from pathlib import Path
import pandas as pd

from ep_pipeline.models import load_embedder, load_spacy_model
from ep_pipeline.scoring.get_td_linguistic import tokenize, load_td_nlp, get_td_metrics_batch
from ep_pipeline.scoring.get_other_linguistic import mattr, windowed_unigram_entropy, trigram_entropy
from ep_pipeline.scoring.get_semantic import compute_semantic_metrics
from ep_pipeline.scoring.get_lexicon import (
    vad_metrics, emotion_metrics, score_lexicon, split_vad_lexicon,
    load_vad_lexicon, load_emotion_lexicon, load_norm_by_name)
from ep_pipeline.scoring.get_syntax_punct import punctuation_metrics, syntax_complexity_metrics
from ep_pipeline.scoring.runner import map_with_checkpoints, map_with_checkpoints_batched


def worker_lingaff_synpun(records_fp, lex_dir, lingaff_fp, synpun_fp, cfg):
    try:
        print("[A] Loading models and lexicons...")
        lex_dir = Path(lex_dir)

        conc_lex = load_norm_by_name(
            lex_dir / "concretness" / "Concreteness_ratings_Brysbaert_et_al_BRM.xlsx",
            term_field="Word", score_field="Conc.M")
        vad_lex_split = emo_lex = aoa_lex = prev_lex = None
        if cfg.affect_mode == "modern":
            vad_lex_split = split_vad_lexicon(
                load_vad_lexicon(lex_dir / "NRC-VAD-Lexicon-v2.1" / "NRC-VAD-Lexicon-v2.1.txt"))
            emo_lex  = load_emotion_lexicon(
                lex_dir / "NRC-Emotion-Intensity-Lexicon" / "NRC-Emotion-Intensity-Lexicon-v1.txt")
            aoa_lex  = load_norm_by_name(
                lex_dir / "AoA" / "AoA_ratings_Kuperman_et_al_BRM_with_PoS.xlsx",
                term_field="Word", score_field="Rating.Mean")
            prev_lex = load_norm_by_name(
                lex_dir / "word_prevelance" / "English_Word_Prevalences.xlsx",
                term_field="Word", score_field="Prevalence")

        nlp_parse = load_spacy_model(cfg.spacy_model, for_tokenizing=False)
        records   = pd.read_parquet(records_fp).to_dict("records")
        print(f"[A] {len(records)} records loaded. Starting lingaff...")

        def score_ling_affect(r):
            toks = tokenize(r["text"], nlp_parse)
            H_mean, H_std, PPL = windowed_unigram_entropy(toks, entropy_window=cfg.entropy_window)
            H3, PPL3 = trigram_entropy(toks, cfg.trigram_test_frac, cfg.trigram_alpha, cfg.seed)
            mattr_score = mattr(toks, cfg.mattr_window)
            out = {
                **r, "id": r["id"], "source": r["source"],
                "mattr": mattr_score,
                "H_unigram_win_mean_nats": H_mean, "H_unigram_win_std_nats": H_std,
                "PPL_unigram_win_mean": PPL, "H_3gram_self_nats": H3, "PPL_3gram_self_nats": PPL3,
            }
            out.update(score_lexicon(toks, conc_lex, prefix="concreteness"))
            if cfg.affect_mode == "modern":
                out.update(vad_metrics(toks, vad_lex_split))
                out.update(emotion_metrics(toks, emo_lex))
                out.update(score_lexicon(toks, aoa_lex, prefix="aoa"))
                out.update(score_lexicon(toks, prev_lex, prefix="prevalence", agg=("mean",)))
            return out

        def score_syntax_punct(r):
            text = r["text"][:cfg.max_text_chars]
            if len(text) > nlp_parse.max_length:
                nlp_parse.max_length = len(text) + 1
            doc = nlp_parse(text)
            out = {**r, "id": r["id"], "source": r["source"]}
            out.update(punctuation_metrics(text))
            out.update(syntax_complexity_metrics(doc))
            return out

        map_with_checkpoints(records, score_ling_affect, Path(lingaff_fp),
                             ["id", "source"], checkpoint_every=cfg.checkpoint_every)
        print("[A] lingaff done. Starting synpun...")
        map_with_checkpoints(records, score_syntax_punct, Path(synpun_fp),
                             ["id", "source"], checkpoint_every=cfg.checkpoint_every)
        print("[A] Done.")

    except Exception:
        print("[A] WORKER FAILED:")
        traceback.print_exc()
        raise


def worker_td(records_fp, td_fp, cfg):
    try:
        print("[B] Loading nlp_td...")
        nlp_td  = load_td_nlp(cfg.spacy_model, cfg.td_metrics)
        records = pd.read_parquet(records_fp).to_dict("records")
        print(f"[B] {len(records)} records loaded. Starting td...")
        map_with_checkpoints_batched(
            records,
            lambda batch: get_td_metrics_batch(batch, nlp_td, cfg.max_text_chars),
            Path(td_fp), ["id", "source"], batch_size=cfg.checkpoint_every)
        print("[B] Done.")

    except Exception:
        print("[B] WORKER FAILED:")
        traceback.print_exc()
        raise


def worker_sem(records_fp, e5_model_path, sem_fp, cfg):
    try:
        print("[C] Loading embedder...")
        embedder = load_embedder(Path(e5_model_path))
        records  = pd.read_parquet(records_fp).to_dict("records")
        print(f"[C] {len(records)} records loaded. Starting sem...")

        def score_semantic(r):
            metrics = compute_semantic_metrics(
                r["text"], embedder,
                chunk_size=cfg.embed_chunk_size, overlap=cfg.embed_overlap,
                batch_size=cfg.batch_size, prefix=cfg.e5_prefix, seed=cfg.seed)
            return {**r, "id": r["id"], "source": r["source"], **metrics}

        map_with_checkpoints(records, score_semantic, Path(sem_fp),
                             ["id", "source"], checkpoint_every=cfg.checkpoint_every)
        print("[C] Done.")

    except Exception:
        print("[C] WORKER FAILED:")
        traceback.print_exc()
        raise
