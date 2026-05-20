from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Dingdong-Inc/FreshRetailNet-50K")
    parser.add_argument("--out", default="data/freshretailnet")
    parser.add_argument("--splits", nargs="+", default=["train", "eval"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        frame = load_dataset(args.dataset, split=split).to_pandas()
        path = out / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        print(path)


if __name__ == "__main__":
    main()
