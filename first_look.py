import argparse
from pathlib import Path

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.adapters.openclip_encoder import OpenClipEncoder
from bioclip_co.application.zero_shot import prompts_for, species_probabilities
from bioclip_co.domain.abstention import descend
from bioclip_co.domain.taxonomy import marginalize


def show_top(title, ranking, n):
    print(f"  {title}")
    for name, p in list(ranking.items())[:n]:
        print(f"    {p:6.1%}  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--candidates", default="data/candidates/trochilidae_colombia.csv")
    # 0.8 is a first guess; it should be tuned on a validation set later
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    taxa = read_candidates(args.candidates)
    print(f"{len(taxa)} candidates")

    for model_name in ["clip", "bioclip"]:
        encoder = OpenClipEncoder(model_name)
        image = encoder.encode_images([args.image])
        texts = encoder.encode_texts(prompts_for(taxa, style=model_name))
        probs = species_probabilities(image, texts, encoder.logit_scale)[0].tolist()

        print(f"\n== {model_name} ==")
        species = {f"{t.common_name} ({t.scientific_name})": p for t, p in zip(taxa, probs)}
        show_top("species", dict(sorted(species.items(), key=lambda x: -x[1])), 5)
        genus = {prefix[-1]: p for prefix, p in marginalize(taxa, probs, "genus").items()}
        show_top("genus", genus, 3)

        pred = descend(taxa, probs, args.threshold)
        hedge = "" if pred.reached_species else "  <- not sure about the species"
        print(f"  answer at {args.threshold:.0%}: {pred.rank} {pred.name} ({pred.confidence:.1%}){hedge}")
