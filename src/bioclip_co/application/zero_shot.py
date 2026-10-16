from collections.abc import Sequence

import numpy as np

from bioclip_co.domain.taxonomy import Taxon


def prompts_for(taxa: Sequence[Taxon], style: str) -> list[str]:
    # The text each model knows best (paper, section 4.1):
    # CLIP was trained on web captions, so common names; BioCLIP on the full taxonomy.
    if style == "clip":
        return [f"a photo of a {t.common_name}" for t in taxa]
    if style == "bioclip":
        return [f"a photo of {t.taxonomic_name} with common name {t.common_name}" for t in taxa]
    raise ValueError(f"unknown prompt style: {style}")


def species_probabilities(image_emb: np.ndarray, text_emb: np.ndarray, logit_scale: float) -> np.ndarray:
    """Softmax over candidate species, one row per image."""
    logits = logit_scale * image_emb @ text_emb.T
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
