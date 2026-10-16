"""The classifier as an MLflow pyfunc model, so a version that passed the quality gate can be registered and served."""
import json

import mlflow
import pandas as pd

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.application.zero_shot import prompts_for, species_probabilities
from bioclip_co.domain.abstention import descend


class BioclipClassifier(mlflow.pyfunc.PythonModel):
    """Input: a DataFrame with an `image` column of file paths. Output: the answer at the calibrated threshold."""

    def load_context(self, context):
        from bioclip_co.adapters.openclip_encoder import OpenClipEncoder

        self.taxa = read_candidates(context.artifacts["candidates"])
        calibration = json.loads(open(context.artifacts["calibration"]).read())["bioclip"]
        self.logit_scale, self.threshold = calibration["logit_scale"], calibration["threshold"]
        self.encoder = OpenClipEncoder("bioclip")
        self.text_emb = self.encoder.encode_texts(prompts_for(self.taxa, style="bioclip"))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        img = self.encoder.encode_images(list(model_input["image"]))
        rows = []
        for probs in species_probabilities(img, self.text_emb, self.logit_scale).tolist():
            pred = descend(self.taxa, probs, self.threshold)
            rows.append({"rank": pred.rank, "name": pred.name, "confidence": pred.confidence})
        return pd.DataFrame(rows)
