import spacy
import torch
from sentence_transformers import SentenceTransformer

# functions for loading models and picking device for computation, which are used in the main pipeline for calculating linguistic and semantic features of texts
def pick_device(): 
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# load a spaCy language model for calculating linguistic features, with an option to disable certain components for faster tokenization when only tokenization is needed
def load_spacy_model(model_name="en_core_web_sm", for_tokenizing=False):
    if for_tokenizing:
        return spacy.load(model_name, disable=["parser", "ner", "lemmatizer", "textcat"])
    return spacy.load(model_name)

# load a sentence embedding model for calculating semantic features based on sentence embeddings, using the SentenceTransformer library and specifying the device for computation
def load_embedder(model_path, device=None):
    device = device or pick_device()
    return SentenceTransformer(model_path, device=device)