import sys

import open_clip
import torch
from PIL import Image

# a few common birds around Pereira
SPECIES = [
    ("Animalia Chordata Aves Apodiformes Trochilidae Amazilia tzacatl", "rufous-tailed hummingbird"),
    ("Animalia Chordata Aves Apodiformes Trochilidae Colibri coruscans", "sparkling violetear"),
    ("Animalia Chordata Aves Passeriformes Thraupidae Sicalis flaveola", "saffron finch"),
    ("Animalia Chordata Aves Passeriformes Tyrannidae Pitangus sulphuratus", "great kiskadee"),
]


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
    path = sys.argv[1]
    clip_prompts = [f"a photo of a {common}" for _, common in SPECIES]
    bio_prompts = [f"a photo of {tax} with common name {common}" for tax, common in SPECIES]

    runs = [
        ("CLIP (OpenAI)", "ViT-B-16", "openai", clip_prompts),
        ("BioCLIP", "hf-hub:imageomics/bioclip", None, bio_prompts),
    ]
    for label, name, pretrained, prompts in runs:
        model, preprocess, tokenizer = load(name, pretrained)
        probs = classify(model, preprocess, tokenizer, path, prompts)
        print(f"\n== {label} ==")
        for (_, common), p in sorted(zip(SPECIES, probs.tolist()), key=lambda x: -x[1]):
            print(f"{p:6.1%}  {common}")
