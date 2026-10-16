import csv
import sys

import open_clip
import torch
from PIL import Image

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
DEFAULT_CANDIDATES = "data/candidates/trochilidae_colombia.csv"


def read_candidates(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(" ".join(row[r] for r in RANKS), row["common_name"].lower()) for row in rows]


def load(name, pretrained=None):
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(name)
    return model.eval(), preprocess, tokenizer


@torch.no_grad()
def classify(model, preprocess, tokenizer, image_path, prompts):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    img = model.encode_image(image)
    txt = model.encode_text(tokenizer(prompts))
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    logits = model.logit_scale.exp() * img @ txt.T
    return logits.softmax(dim=-1)[0]


if __name__ == "__main__":
    image_path = sys.argv[1]
    species = read_candidates(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CANDIDATES)

    clip_prompts = [f"a photo of a {common}" for _, common in species]
    bio_prompts = [f"a photo of {tax} with common name {common}" for tax, common in species]

    runs = [
        ("CLIP (OpenAI)", "ViT-B-16", "openai", clip_prompts),
        ("BioCLIP", "hf-hub:imageomics/bioclip", None, bio_prompts),
    ]
    print(f"{len(species)} candidates")
    for label, name, pretrained, prompts in runs:
        model, preprocess, tokenizer = load(name, pretrained)
        probs = classify(model, preprocess, tokenizer, image_path, prompts)
        print(f"\n== {label} ==")
        ranked = sorted(zip(species, probs.tolist()), key=lambda x: -x[1])
        for (tax, common), p in ranked[:5]:
            print(f"{p:6.1%}  {common:<28} ({' '.join(tax.split()[-2:])})")
