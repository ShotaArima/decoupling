from __future__ import annotations

import argparse
import json

from .evaluate import run_evaluation
from .train import train_glr


def main() -> None:
    parser = argparse.ArgumentParser(description="Decoupled local/global representations for FreshRetailNet.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/freshretailnet.json")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", default="runs/freshretailnet_glr/best.pt")
    args = parser.parse_args()

    if args.cmd == "train":
        print(train_glr(args.config))
    elif args.cmd == "evaluate":
        print(json.dumps(run_evaluation(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
