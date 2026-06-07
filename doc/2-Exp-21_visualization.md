# 2-Exp-21: FreshRetailNet Residual Visualization

## 背景

2-Exp-20 では、`series_mean_all` において baseline 改善の信頼区間が 0 未満になった。

ただし、論文では数値表だけでは不十分である。読者に伝えるべきことは次の 3 つ。

1. `series_mean` residual には hour profile が残る。
2. 提案モデルの hour component は、その residual hour profile に対応する。
3. `same_hour_recent_mean` のような強い baseline では、残差構造が薄くなり、改善余地が小さい。

## 目的

2-Exp-21 では、論文図を作るための軽量 CSV を出力する。

対象は次の 2 scenario。

| scenario | 目的 |
| --- | --- |
| `series_mean_all` | 成功例。hour 構造と補正改善を見る |
| `same_hour_recent_mean_d7_all` | 限界例。強い baseline 後に構造が薄いことを見る |

比較 variant は次の 2 つ。

| variant | 目的 |
| --- | --- |
| `centered_raw` | calibration なしの成分分解 |
| `bias_constrained_001` | 2-Exp-20 で high residual と bias の trade-off が良かったモデル |

## 出力

各 variant の `visualization/` に次を出す。

| file | 内容 |
| --- | --- |
| `profiles_by_hour.csv` | residual / residual_hat / component / error の hour 平均 |
| `profiles_by_day.csv` | residual / residual_hat / component / error の day 平均 |
| `series_summary.csv` | series ごとの baseline MAE、corrected MAE、改善量 |
| `series_best_01.csv` など | 補正が効いた series の cell-level 表 |
| `series_worst_01.csv` など | 補正が悪化した series の cell-level 表 |

これらを使って、heatmap や hour profile line plot を作る。

## 良い結果

- `series_mean_all` の `profiles_by_hour.csv` で、`residual` と `hour_component` の形が近い。
- `series_mean_all` の `series_best_*.csv` で、高残差の時間帯に補正が入っている。
- `same_hour_recent_mean_d7_all` では `hour_component` が小さく、改善量も小さい。

## 悪い結果

- `series_mean_all` でも `hour_component` と `residual` の profile が合わない。
- best series が単なる平均補正に見える。
- worst series で特定の欠品やゼロ売上に強く引っ張られている。

## 実行コマンド

smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-21_visualization_smoke.json
```

本実験:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-21_freshretailnet_visualization.json
```

## 論文上の使い方

2-Exp-21 の図は、FreshRetailNet 節の qualitative analysis に使う。

数値主張は 2-Exp-19/20 で支え、2-Exp-21 は「何を拾っているのか」を説明するための補助図にする。
