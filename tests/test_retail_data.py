from __future__ import annotations

import unittest
from unittest.mock import patch

from decoupled_ts.retail_data import build_retail_data


class FreshRetailNetSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "seed": 17,
            "dataset": {
                "name": "freshretailnet",
                "train_split": "train",
                "eval_split": "eval",
                "history_days": 28,
                "forecast_days": 2,
                "max_train_series": 2000,
                "max_valid_series": 500,
                "max_eval_series": 500,
                "daily_numeric_columns": [],
                "subset_filter": {"enabled": False},
            },
        }

    @patch("decoupled_ts.retail_data.infer_input_dim", return_value=6)
    @patch("decoupled_ts.retail_data.FreshRetailNetSeries.from_config")
    def test_train_holdout_uses_disjoint_train_range_and_eval_test(self, from_config, _infer_input_dim) -> None:
        train_pool = list(range(2500))
        test = object()
        from_config.side_effect = [train_pool, test]
        self.config["dataset"]["validation_source"] = "train_holdout"

        bundle = build_retail_data(self.config)

        self.assertEqual(list(bundle.train.indices), list(range(2000)))
        self.assertEqual(list(bundle.valid.indices), list(range(2000, 2500)))
        self.assertIs(bundle.train.dataset, train_pool)
        self.assertIs(bundle.valid.dataset, train_pool)
        self.assertIs(bundle.test, test)
        calls = from_config.call_args_list
        self.assertEqual((calls[0].args[1], calls[0].kwargs["max_series"]), ("train", 2500))
        self.assertEqual(calls[0].args[0]["series_start_offset"], 0)
        self.assertEqual((calls[1].args[1], calls[1].kwargs["max_series"]), ("eval", 500))

    @patch("decoupled_ts.retail_data.infer_input_dim", return_value=6)
    @patch("decoupled_ts.retail_data.FreshRetailNetSeries.from_config")
    def test_overlapping_holdout_ranges_are_rejected(self, from_config, _infer_input_dim) -> None:
        from_config.return_value = object()
        self.config["dataset"].update(
            {
                "validation_source": "train_holdout",
                "validation_series_start_offset": 1900,
            }
        )

        with self.assertRaisesRegex(ValueError, "overlap"):
            build_retail_data(self.config)


if __name__ == "__main__":
    unittest.main()
