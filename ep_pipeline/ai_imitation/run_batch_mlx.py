import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from ep_pipeline.ai_imitation.completion_mlx import CompletionMLX

def run_batch(
    prompts_file: Path,
    output_file: Path,
    model_id: str,
    max_new_tokens: int = 650,
    temp: float = 0.7,
    top_p: float = 0.9,
) -> None:
    completed = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["id"])
                except Exception:
                    pass

    print(f"Resuming — {len(completed)} already completed")

    model = CompletionMLX(
        model_id=model_id,
        sampling_params={"temp": temp, "top_p": top_p},
    )
    print("Loading model (first run may take ~1 minute)...")
    model.load()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with prompts_file.open("r", encoding="utf-8") as fin, \
         output_file.open("a", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)
            uid = item["id"]

            if uid in completed:
                continue

            messages = [{"role": "user", "content": item["prompt"]}]
            prompt = model.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            text = model.generate(prompt, max_new_tokens=max_new_tokens)

            record = {
                "id":             uid,
                "completion":     text,
                "model":          model_id,
                "max_new_tokens": max_new_tokens,
                "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"✔ {uid}")

    print("Finished batch.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLX batch completions")
    parser.add_argument("--prompts",        required=True, type=Path,  help="Path to .jsonl prompts file")
    parser.add_argument("--output",         required=True, type=Path,  help="Path to .jsonl output file")
    parser.add_argument("--model",          required=True, type=str,   help="Model ID or path")
    parser.add_argument("--max-new-tokens", default=650,   type=int,   help="Max tokens to generate")
    parser.add_argument("--temp",           default=0.7,   type=float, help="Sampling temperature")
    parser.add_argument("--top-p",          default=0.9,   type=float, help="Top-p sampling")
    args = parser.parse_args()

    run_batch(args.prompts, args.output, args.model,
              max_new_tokens=args.max_new_tokens, temp=args.temp, top_p=args.top_p)


if __name__ == "__main__":
    main()
