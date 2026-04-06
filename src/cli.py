from __future__ import annotations

import argparse
import json

from .agent import HateSpeechAgent
from .config import DEFAULT_DATASET, DEFAULT_MODEL_PATH, ensure_directories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local hate speech detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    evaluate_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))

    predict_parser = subparsers.add_parser("predict", help="Predict hate speech")
    predict_parser.add_argument("--text", action="append", required=True)
    predict_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))

    return parser


def main() -> None:
    ensure_directories()
    parser = build_parser()
    args = parser.parse_args()
    agent = HateSpeechAgent()

    if args.command == "train":
        metrics = agent.train(dataset_path=args.dataset, model_path=args.model)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "evaluate":
        metrics = agent.evaluate(dataset_path=args.dataset, model_path=args.model)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "predict":
        results = agent.predict(args.text, model_path=args.model)
        print(json.dumps(results, indent=2))
        return

