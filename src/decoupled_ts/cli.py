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
    retail = sub.add_parser("retail-experiment")
    retail.add_argument("--config", default="configs/retail_multigrain.json")
    residual = sub.add_parser("residual-diagnostics")
    residual.add_argument("--config", default="configs/2-Exp-1_residual_diagnostics_smoke.json")
    residual_exp = sub.add_parser("residual-experiment")
    residual_exp.add_argument("--config", default="configs/2-Exp-2_to_6_residual_smoke.json")
    residual_sweep = sub.add_parser("residual-sweep")
    residual_sweep.add_argument("--config", default="configs/2-Exp-9_multiseed_structured_residual_smoke.json")
    same_hour = sub.add_parser("same-hour-analysis")
    same_hour.add_argument("--config", default="configs/EXP-008_same_hour_analysis_freshretailnet.json")
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
    elif args.cmd == "retail-experiment":
        from .retail_experiments import run_retail_experiments

        print(json.dumps(run_retail_experiments(args.config), indent=2))
    elif args.cmd == "residual-diagnostics":
        from .residual_diagnostics import run_residual_diagnostics

        print(json.dumps(run_residual_diagnostics(args.config), indent=2))
    elif args.cmd == "residual-experiment":
        from .residual_experiments import run_residual_experiments

        print(json.dumps(run_residual_experiments(args.config), indent=2))
    elif args.cmd == "residual-sweep":
        from .residual_sweeps import run_residual_sweep

        print(json.dumps(run_residual_sweep(args.config), indent=2))
    elif args.cmd == "same-hour-analysis":
        from .retail_analysis import run_same_hour_analysis

        print(json.dumps(run_same_hour_analysis(args.config), indent=2))


if __name__ == "__main__":
    main()
