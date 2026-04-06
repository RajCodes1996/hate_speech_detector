from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import (
    BORDERLINE_MARGIN,
    DEFAULT_DATASET,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    PREDICTION_THRESHOLD,
    RANDOM_STATE,
    TEST_SIZE,
)
from .data import load_dataset, split_features_labels
from .text_cleaning import clean_text
from .utils import save_json


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                preprocessor=clean_text,
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=1,
                                sublinear_tf=True,
                                max_features=12_000,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                preprocessor=clean_text,
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                                sublinear_tf=True,
                                max_features=12_000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _best_threshold(y_true, probabilities) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-9
    )
    best_index = int(f1_scores.argmax())
    return float(thresholds[best_index])


def _extract_model(bundle_or_model):
    if isinstance(bundle_or_model, dict) and "model" in bundle_or_model:
        return bundle_or_model["model"], float(bundle_or_model.get("threshold", PREDICTION_THRESHOLD))
    return bundle_or_model, PREDICTION_THRESHOLD


@dataclass
class HateSpeechAgent:
    model: Pipeline | None = None
    threshold: float = 0.5

    def train(
        self,
        dataset_path: str | Path = DEFAULT_DATASET,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metrics_path: str | Path = DEFAULT_METRICS_PATH,
    ) -> dict:
        df = load_dataset(dataset_path)
        if len(df) < 10:
            raise ValueError("Dataset is too small. Add more examples before training.")
        if df["label"].value_counts().min() < 2:
            raise ValueError(
                "Each class needs at least 2 examples for a stratified train/test split."
            )

        X, y = split_features_labels(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        self.model = build_pipeline()
        min_class_size = int(y_train.value_counts().min())
        cv_folds = max(2, min(5, min_class_size))
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        oof_probabilities = cross_val_predict(
            self.model,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        self.threshold = _best_threshold(y_train, oof_probabilities)
        self.model.fit(X_train, y_train)

        test_probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (test_probabilities >= self.threshold).astype(int)
        metrics = self._build_metrics(y_test, predictions)
        metrics["threshold"] = self.threshold
        self.save_model(model_path)
        save_json(metrics, metrics_path)
        return metrics

    def evaluate(
        self,
        dataset_path: str | Path = DEFAULT_DATASET,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> dict:
        bundle = self.load_model(model_path)
        self.model, self.threshold = _extract_model(bundle)
        df = load_dataset(dataset_path)
        if len(df) < 10:
            raise ValueError("Dataset is too small. Add more examples before evaluating.")
        if df["label"].value_counts().min() < 2:
            raise ValueError(
                "Each class needs at least 2 examples for a stratified train/test split."
            )
        X, y = split_features_labels(df)
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        metrics = self._build_metrics(y_test, predictions)
        metrics["threshold"] = self.threshold
        return metrics

    def predict(
        self,
        texts: Iterable[str],
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> list[dict]:
        bundle = self.load_model(model_path)
        model, threshold = _extract_model(bundle)
        raw_texts = list(texts)
        cleaned_texts = [clean_text(text) for text in raw_texts]

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(cleaned_texts)[:, 1]
        else:
            predictions = model.predict(cleaned_texts)
            probabilities = [float(value) for value in predictions]

        results = []
        for index, text in enumerate(raw_texts):
            probability = float(probabilities[index])
            is_hate = probability >= threshold
            distance = abs(probability - threshold)
            item = {
                "text": text,
                "label": int(is_hate),
                "prediction": "hate_speech" if is_hate else "not_hate",
                "hate_probability": probability,
                "threshold": threshold,
                "borderline": distance <= BORDERLINE_MARGIN,
            }
            if item["borderline"]:
                item["note"] = "Borderline prediction: the score is close to the cutoff."
            elif is_hate:
                item["note"] = "Above threshold, so it is flagged as hate speech."
            else:
                item["note"] = "Below threshold, so it is treated as not hate speech."
            results.append(item)
        return results

    def save_model(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        if self.model is None:
            raise ValueError("No trained model available to save.")
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold}, model_path)

    def load_model(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Train the model first."
            )
        return joblib.load(model_path)

    @staticmethod
    def _build_metrics(y_true, y_pred) -> dict:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": report,
        }
