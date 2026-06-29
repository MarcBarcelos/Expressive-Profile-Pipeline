import pandas as pd
from ep_pipeline.io import read_checkpoint, write_table

def _key(record, key_cols):
    return "__".join(str(record[c]) for c in key_cols)

def map_with_checkpoints(records, score_fn, checkpoint_path, key_cols, checkpoint_every=50, on_error="skip"):
    done = read_checkpoint(checkpoint_path)
    done_rows = done.to_dict("records") if done is not None else []
    done_keys = {_key(r, key_cols) for r in done_rows} if done is not None else set()

    results = list(done_rows)
    todo = [r for r in records if _key(r, key_cols) not in done_keys]

    last_flush = -1
    for i, record in enumerate(todo):
        try:
            out = score_fn(record)
        except Exception as e:
            print(f"  [FAIL] {_key(record, key_cols)}: {e}")
            if on_error == "raise":
                raise
            out = {c: record[c] for c in key_cols} if on_error == "record" else None

        if out is not None:
            results.append(out)
        if (i + 1) % checkpoint_every == 0:
            write_table(pd.DataFrame(results), checkpoint_path)
            last_flush = i
            print(f"  checkpoint: {i + 1}/{len(todo)}")

    final = pd.DataFrame(results)
    if last_flush != len(todo) - 1:
        write_table(final, checkpoint_path)
    print(f"Done: {len(final)} rows -> {checkpoint_path}")
    return final

def map_with_checkpoints_batched(records, batch_fn, checkpoint_path, key_cols, batch_size=50):
    done = read_checkpoint(checkpoint_path)
    done_rows = done.to_dict("records") if done is not None else []
    done_keys = {_key(r, key_cols) for r in done_rows} if done is not None else set()

    results = list(done_rows)
    todo = [r for r in records if _key(r, key_cols) not in done_keys]

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        results.extend(batch_fn(batch))
        write_table(pd.DataFrame(results), checkpoint_path)
        print(f"  checkpoint: {min(i + batch_size, len(todo))}/{len(todo)}")

    final = pd.DataFrame(results)
    if not todo:
        write_table(final, checkpoint_path)
    print(f"Done: {len(final)} rows -> {checkpoint_path}")
    return final
