import textdescriptives as td

# tokenize the input text using the provided spaCy language model, which is used for calculating linguistic features
def tokenize(text, nlp):
    if len(text) > nlp.max_length:                              # if length of the input text exceeds maximum length allowed by the spaCy model, set maximum length to one more than the length of input text
        nlp.max_length = len(text) + 1
    doc = nlp.make_doc(text)                                    # create a spaCy Doc object from the input text which allows for tokenization and processing of the text for calculating linguistic features
    return [t.text.lower() for t in doc if not t.is_space]      # return a list of lowercase tokens from the Doc object, excluding any tokens that are just whitespace

# calculate the linguistic metrics for the input text using the provided spaCy language model and specified metrics
def get_td_metrics(text, spacy_model, td_metrics):
    df = td.extract_metrics(text, spacy_model=spacy_model, metrics=td_metrics) # calculate the specified linguistic metrics for the input text using the textdescriptives library
    row = df.iloc[0].to_dict()                                                 # extract the first row of the resulting DataFrame as a dictionary, which contains the calculated linguistic metrics 
    row.pop("text", None)                                                      # remove the "text" key from the dictionary if it exists, as it is not needed for the final output of linguistic features
    return row                                                                 # return the dictionary containing the calculated linguistic metrics for the input text