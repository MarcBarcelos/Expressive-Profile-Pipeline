import pandas as pd
from ep_pipeline.io import read_checkpoint, write_table

# Helper functions for applying a scoring function to a list of records with checkpointing to avoid redundant computation and allow resuming from intermediate results in case of failure or interruption.
def _key(record, key_cols):
    return "__".join(str(record[c]) for c in key_cols)

# map a scoring function over a list of records with checkpointing to save intermediate results and allow resuming from checkpoints in case of failure or interruption
def map_with_checkpoints(records, score_fn, checkpoint_path, key_cols, checkpoint_every=50, on_error="skip"): 
    """ Apply `score_fn` to each record in `records`, but save intermediate results to `checkpoint_path` every `checkpoint_every` records, and load from the checkpoint if it exists to avoid redundant computation
    
    records: list of dicts or a dataframe to process (eg. chunks_df.to_dict("records"))
    score_fn: function that takes a record and returns a dict of results to be saved
    key_cols: list of columns to use as a unique key for each record, used for checkpointing
    """
    done = read_checkpoint(checkpoint_path)                                             # read existing checkpoint if it exists to determine which records have already been processed and avoid redundant computation
    done_rows = done.to_dict("records") if done is not None else []                     # convert the existing checkpoint to a list of records (dictionaries) if it exists, otherwise use an empty list to indicate that no records have been processed yet
    done_keys = {_key(r, key_cols) for r in done_rows} if done is not None else set()   # create a set of keys for the records that have already been processed based on the existing checkpoint, which allows for efficient lookup to determine if a record has already been processed when iterating through the list of records to apply the scoring function

    results = list(done_rows)                                           # initialize the results list with the records from the existing checkpoint
    todo = [r for r in records if _key(r, key_cols) not in done_keys]   # create a list of records that still need to be processed by filtering out the records that have already been processed based on the keys in the existing checkpoint

    for i, record in enumerate(todo): # iterate through the list of records that still need to be processed, applying the scoring function to each record and handling any exceptions that may occur during processing based on the specified error handling strategy (on_error)
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
            print(f"  checkpoint: {i + 1}/{len(todo)}")
 
    final = pd.DataFrame(results)
    write_table(final, checkpoint_path)   # final flush captures the tail
    print(f"Done: {len(final)} rows -> {checkpoint_path}")
    return final