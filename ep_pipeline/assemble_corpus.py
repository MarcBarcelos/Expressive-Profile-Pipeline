import pandas as pd

def extract_passage(prompt, marker="PASSAGE:"):
    idx = prompt.find(marker)
    return prompt[idx + len(marker):].strip() if idx != -1 else prompt.strip()

def build_text_table(records, source="human", *, id_key="id", text_key="text",
                     prompt_key=None, marker="PASSAGE:",
                     outputs=None, completion_key="completion", output_source="ai"):
    out_by_id = {o[id_key]: o[completion_key] for o in (outputs or [])}
    rows = []
    for r in records:
        uid = r[id_key]
        text = extract_passage(r[prompt_key], marker) if prompt_key else r[text_key]
        rows.append({"id": uid, "source": source, "text": text})
        if uid in out_by_id:                      # only fires in prompt+output mode
            rows.append({"id": uid, "source": output_source, "text": out_by_id[uid]})
    return pd.DataFrame(rows)

# Full text        -> build_text_table(docs)
# Prompt + AI       -> build_text_table(prompts, prompt_key="prompt", outputs=outputs)