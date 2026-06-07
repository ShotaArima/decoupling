# 2-Exp-20: Statistical Validation of FreshRetailNet Residual Correction

## 背景

2-Exp-19 では、bias 制約つき calibration が validation 上の bias 制約を満たした。

結果の読み方は単純ではない。

- `mae_grid_reference` は全体 MAE が最も良い。
- `bias_constrained_001` は bias を大きく下げ、`series_mean_all` の high residual top10 で最も良い。
- `affine_reference` は bias を消せるが、全体 MAE が悪化する。

論文に載せるには、平均値だけでなく seed 間で安定しているかを見る必要がある。

## 目的

2-Exp-20 は、2-Exp-19 の 5 seed 結果に対して seed-level paired bootstrap を行う。

目的は次の 3 つ。

1. baseline に対する改善が seed 平均だけでなく安定しているか。
2. `bias_constrained_001` が `mae_grid_reference` より bias / high residual で良いと言えるか。
3. `bias_constrained_001` が `affine_reference` より全体 MAE を保てるか。

## 実装

追加 CLI:

```bash
uv run decoupled-ts residual-result-analysis --config configs/2-Exp-20_statistical_validation.json
```

入力は `runs/2-Exp-19_freshretailnet_bias_constrained_calibration/summary.json`。

出力:

| file | 内容 |
| --- | --- |
| `baseline_delta.csv` | 各 model が baseline よりどれだけ良いか |
| `paired_model_delta.csv` | model 間の seed 対応差 |
| `summary.json` | 上記の JSON 版 |

## 良い結果

- `series_mean_all / bias_constrained_001 / calibrated_corrected_cell_mae` の delta CI が 0 未満。
- `series_mean_all / bias_constrained_001 / calibrated_high_residual_top10_corrected_mae` の delta CI が 0 未満。
- `bias_constrained_001 - affine_reference` の `calibrated_corrected_cell_mae` が 0 未満。
- `bias_constrained_001 - mae_grid_reference` の `calibrated_high_residual_top10_corrected_mae` が 0 未満。

## 悪い結果

- baseline に対する CI が 0 をまたぐ。
- `bias_constrained_001` が high residual でも `mae_grid_reference` に勝たない。
- `affine_reference` と比べて MAE 優位がない。

## 実行コマンド

smoke:

```bash
uv run decoupled-ts residual-result-analysis --config configs/2-Exp-20_statistical_validation_smoke.json
```

本実験:

```bash
uv run decoupled-ts residual-result-analysis --config configs/2-Exp-20_statistical_validation.json
```

## 論文上の判断

2-Exp-20 で baseline 改善の信頼区間が 0 未満なら、FreshRetailNet でも「条件付きで予測補正として有効」と書ける。

もし CI が弱い場合でも、次の主張に落とせる。

```text
FreshRetailNet では series_mean residual に hour 構造が残り、提案分解はその構造を安定して抽出する。
ただし calibration の目的関数により、全体 MAE、bias、高残差改善の間には trade-off がある。
```
