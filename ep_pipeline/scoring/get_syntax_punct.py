import numpy as np

# Punctuation / orthographic expressivity  (operates on raw text)
def punctuation_metrics(text, n_tokens=None):
    out = {}
    n_chars = len(text) if text else 0
    if n_chars == 0:
        keys = ["comma_rate", "semicolon_rate", "colon_rate", "dash_rate",
                "exclamation_rate", "question_rate", "parenthesis_rate",
                "quotation_rate", "ellipsis_rate"]
        for k in keys:
            out[k] = float("nan")
        return out
    per1k = 1000.0 / n_chars
    out["comma_rate"]       = text.count(",") * per1k
    out["semicolon_rate"]   = text.count(";") * per1k
    out["colon_rate"]       = text.count(":") * per1k
    out["dash_rate"]        = (text.count("—") + text.count("–") + text.count(" - ")) * per1k   # em/en/spaced hyphen as dash
    out["exclamation_rate"] = text.count("!") * per1k
    out["question_rate"]    = text.count("?") * per1k
    out["parenthesis_rate"] = text.count("(") * per1k                                           # count opening parens only (pairs)
    out["quotation_rate"]   = (text.count('"') + text.count("“") + text.count("”")) * per1k
    out["ellipsis_rate"]    = (text.count("…") + text.count("...")) * per1k
    if n_tokens:                                                                                # optional, only if casing is preserved upstream
        caps = sum(1 for w in text.split() if len(w) > 1 and w.isupper())
        out["all_caps_word_rate"] = float(caps / n_tokens) if n_tokens else float("nan")
    return out

# Deeper syntactic complexity  (operates on a parsed spaCy Doc)
# Universal Dependencies clause relations used to count subordinate clauses.
_SUBORD_DEPS = {"advcl", "ccomp", "xcomp", "acl", "relcl", "csubj", "csubjpass", "acl:relcl"}
_COORD_DEPS  = {"conj", "cc"}
_CLAUSE_HEAD_POS = {"VERB", "AUX"}                                  # finite-clause proxy: count verbal heads

def _tree_depth(token):
    """Max dependency-tree depth below a token (root -> deepest leaf)."""
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(_tree_depth(c) for c in children)

def syntax_complexity_metrics(doc):
    sents = list(doc.sents)
    keys = ["subordination_ratio", "coordination_ratio", "clauses_per_sentence",
            "parse_tree_depth_mean", "parse_tree_depth_std",
            "mean_dependents_per_head", "prop_complex_sentences"]
    if not sents:
        return {k: float("nan") for k in keys}
    n_sents = len(sents)
    subord = coord = clauses = 0
    depths, complex_sents = [], 0
    n_heads_with_children, n_dependents = 0, 0
    for sent in sents:
        sent_subord = 0
        root = sent.root
        depths.append(_tree_depth(root))
        for tok in sent:
            dep = tok.dep_.lower()
            if dep in _SUBORD_DEPS:
                subord += 1
                sent_subord += 1
            if dep in _COORD_DEPS:
                coord += 1
            if tok.pos_ in _CLAUSE_HEAD_POS:
                clauses += 1
            n_kids = sum(1 for _ in tok.children)
            if n_kids:
                n_heads_with_children += 1
                n_dependents += n_kids
        if sent_subord >= 1:
            complex_sents += 1
    depths = np.asarray(depths, dtype=float)
    return {
        "subordination_ratio":      float(subord / n_sents),
        "coordination_ratio":       float(coord / n_sents),
        "clauses_per_sentence":     float(clauses / n_sents),
        "parse_tree_depth_mean":    float(depths.mean()),
        "parse_tree_depth_std":     float(depths.std()),
        "mean_dependents_per_head": float(n_dependents / n_heads_with_children) if n_heads_with_children else float("nan"),
        "prop_complex_sentences":   float(complex_sents / n_sents),
    }
