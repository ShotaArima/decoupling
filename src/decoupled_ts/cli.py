from __future__ import annotations

import argparse
import json
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Decoupled local/global representations for FreshRetailNet.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/freshretailnet.json")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", default="runs/freshretailnet_glr/best.pt")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.cmd == "train":
        from .train import train_glr

        print(train_glr(args.config))
    elif args.cmd == "evaluate":
        from .evaluate import run_evaluation

        print(json.dumps(run_evaluation(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
