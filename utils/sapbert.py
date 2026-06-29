
import numpy as np

SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

_tokenizer = None
_model = None
_cache = {}


def load_sapbert():
    """
    Loads SapBERT tokenizer and model on first call.
    Returns cached versions on subsequent calls.
    """
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    print("[SapBERT] Loading model (first use - ~15s)...")

    from transformers import AutoTokenizer, AutoModel

    _tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL_NAME)
    _model = AutoModel.from_pretrained(SAPBERT_MODEL_NAME)
    _model.eval()

    print("[SapBERT] Model loaded.")
    return _tokenizer, _model


def get_embedding(text: str) -> np.ndarray:
    """
    Embeds a clinical term into a 768-dim vector.
    Uses mean pooling across all token embeddings.
    Result is cached — same term is never embedded twice.
    """
    if text in _cache:
        return _cache[text]

    import torch

    tokenizer, model = load_sapbert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = (
        outputs.last_hidden_state
        .mean(dim=1)
        .squeeze()
        .numpy()
    )

    _cache[text] = embedding
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Returns float in [-1, 1]. Closer to 1 = more similar.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))