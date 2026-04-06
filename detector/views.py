from __future__ import annotations

from pathlib import Path
import json

from django.contrib import messages
from django.shortcuts import render

from src.agent import HateSpeechAgent
from src.config import DEFAULT_DATASET, DEFAULT_MODEL_PATH, ensure_directories

from .forms import DatasetUploadForm, PredictionForm

agent = HateSpeechAgent()


def _load_status():
    model_exists = Path(DEFAULT_MODEL_PATH).exists()
    dataset_exists = Path(DEFAULT_DATASET).exists()
    return {
        "model_exists": model_exists,
        "dataset_exists": dataset_exists,
    }


def index(request):
    ensure_directories()
    train_form = DatasetUploadForm()
    predict_form = PredictionForm()
    results = None
    metrics = None
    status = _load_status()

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "train":
            train_form = DatasetUploadForm(request.POST, request.FILES)
            if train_form.is_valid():
                uploaded_file = train_form.cleaned_data["dataset"]
                dataset_path = Path(DEFAULT_DATASET)
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                with dataset_path.open("wb+") as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                try:
                    metrics = agent.train(dataset_path=dataset_path, model_path=DEFAULT_MODEL_PATH)
                    messages.success(request, "Model trained successfully.")
                    status = _load_status()
                except Exception as exc:
                    messages.error(request, f"Training failed: {exc}")
            else:
                messages.error(request, "Please upload a valid CSV file.")

        elif action == "predict":
            predict_form = PredictionForm(request.POST)
            if predict_form.is_valid():
                raw_text = predict_form.cleaned_data["text"]
                texts = [line.strip() for line in raw_text.splitlines() if line.strip()]
                if not texts:
                    messages.error(request, "Please enter at least one sentence to analyze.")
                else:
                    try:
                        results = agent.predict(texts, model_path=DEFAULT_MODEL_PATH)
                    except Exception as exc:
                        messages.error(request, f"Prediction failed: {exc}")
            else:
                messages.error(request, "Please enter some text to analyze.")

    return render(
        request,
        "detector/index.html",
        {
            "train_form": train_form,
            "predict_form": predict_form,
            "results": results,
            "metrics": json.dumps(metrics, indent=2) if metrics else None,
            "status": status,
        },
    )
