import numpy as np


def nll(image_emb: np.ndarray, text_emb: np.ndarray, truth: np.ndarray, logit_scale: float) -> float:
    """Mean negative log-likelihood of the true species."""
    logits = logit_scale * image_emb @ text_emb.T
    logits -= logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(truth)), truth].mean())


def fit_logit_scale(image_emb: np.ndarray, text_emb: np.ndarray, truth: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): the logit scale that best predicts the true species.

    It doesn't change which species ranks first, only how confident the probabilities are.
    A grid is enough for one parameter and keeps it easy to read.
    """
    scales = np.geomspace(1, 300, 400)
    return float(min(scales, key=lambda s: nll(image_emb, text_emb, truth, s)))
