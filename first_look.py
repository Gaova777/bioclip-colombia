import argparse

import open_clip
import torch
from PIL import Image

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.domain.abstention import descend
from bioclip_co.domain.taxonomy import marginalize


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
    return logits.softmax(dim=-1)[0].tolist()


def show_top(title, ranking, n):
    print(f"  {title}")
    for name, p in list(ranking.items())[:n]:
        print(f"    {p:6.1%}  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--candidates", default="data/candidates/trochilidae_colombia.csv")
    # 0.8 is a first guess; it should be tuned on a validation set later
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    image_path = args.image
    taxa = read_candidates(args.candidates)

    clip_prompts = [f"a photo of a {t.common_name}" for t in taxa]
    bio_prompts = [f"a photo of {t.taxonomic_name} with common name {t.common_name}" for t in taxa]

    runs = [
        ("CLIP (OpenAI)", "ViT-B-16", "openai", clip_prompts),
        ("BioCLIP", "hf-hub:imageomics/bioclip", None, bio_prompts),
    ]
    print(f"{len(taxa)} candidates")
    for label, name, pretrained, prompts in runs:
        model, preprocess, tokenizer = load(name, pretrained)
        probs = classify(model, preprocess, tokenizer, image_path, prompts)
        print(f"\n== {label} ==")
        species = {f"{t.common_name} ({t.scientific_name})": p for t, p in zip(taxa, probs)}
        show_top("species", dict(sorted(species.items(), key=lambda x: -x[1])), 5)
        genus = {prefix[-1]: p for prefix, p in marginalize(taxa, probs, "genus").items()}
        show_top("genus", genus, 3)

        pred = descend(taxa, probs, args.threshold)
        hedge = "" if pred.reached_species else "  <- not sure about the species"
        print(f"  answer at {args.threshold:.0%}: {pred.rank} {pred.name} ({pred.confidence:.1%}){hedge}")
