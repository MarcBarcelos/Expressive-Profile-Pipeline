import spacy
import textdescriptives as td

# tokenize the input text using the provided spaCy language model, which is used for calculating linguistic features
def tokenize(text, nlp):
    if len(text) > nlp.max_length:                              # if length of the input text exceeds maximum length allowed by the spaCy model, set maximum length to one more than the length of input text
        nlp.max_length = len(text) + 1
    doc = nlp.make_doc(text)                                    # create a spaCy Doc object from the input text which allows for tokenization and processing of the text for calculating linguistic features
    return [t.text.lower() for t in doc if not t.is_space]      # return a list of lowercase tokens from the Doc object, excluding any tokens that are just whitespace

# calculate the linguistic metrics for the input text using the provided spaCy language model and specified metrics
def load_td_nlp(spacy_model, td_metrics):
    nlp = spacy.load(spacy_model, exclude=["ner"])
    for metric in td_metrics:
        nlp.add_pipe(f"textdescriptives/{metric}")
    return nlp

def get_td_metrics(text, nlp, max_chars=999_999):
    if len(text) > nlp.max_length:
        nlp.max_length = len(text) + 1
    if len(text) > max_chars:
        text = text[:max_chars]
    doc = nlp(text)
    df = td.extract_df(doc, include_text=False)
    row = df.iloc[0].to_dict()
    row.pop("text", None)
    return row

def get_td_metrics_batch(records, nlp, max_chars=999_999):
    texts = [r["text"][:max_chars] for r in records]
    max_len = max(len(t) for t in texts)
    if max_len > nlp.max_length:
        nlp.max_length = max_len + 1
    results = []
    for r, doc in zip(records, nlp.pipe(texts)):
        try:
            row = td.extract_df(doc, include_text=False).iloc[0].to_dict()
            row.pop("text", None)
            results.append({**r, **row})
        except Exception as e:
            print(f"  [FAIL] {r['id']}: {e}")
    return results
