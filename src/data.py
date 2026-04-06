from __future__ import annotations

from pathlib import Path

import pandas as pd

from .text_cleaning import clean_text


HATE_LABELS = {"1", "hate", "hateful", "offensive", "toxic", "abusive", "spam"}
NON_HATE_LABELS = {"0", "not_hate", "normal", "clean", "neutral", "safe"}


def normalize_label(value) -> int:
    if pd.isna(value):
        raise ValueError("Found an empty label in the dataset.")

    if isinstance(value, (int, float)):
        if int(value) in (0, 1):
            return int(value)

    label = str(value).strip().lower()
    if label in HATE_LABELS:
        return 1
    if label in NON_HATE_LABELS:
        return 0

    try:
        numeric_label = int(float(label))
        if numeric_label in (0, 1):
            return numeric_label
    except ValueError as exc:
        raise ValueError(f"Unsupported label value: {value!r}") from exc

    raise ValueError(f"Unsupported label value: {value!r}")


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing))}"
        )

    df = df.loc[:, ["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label"] = df["label"].map(normalize_label)
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def split_features_labels(df: pd.DataFrame):
    return df["text"], df["label"]

