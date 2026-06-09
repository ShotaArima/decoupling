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

## 実行結果

添付された標準出力では、4 条件の metrics が確認できた。

| scenario | model | baseline MAE | corrected MAE | calibrated MAE | baseline top10 | corrected top10 | calibrated top10 | bias | calibrated bias | hour delta | hour corr | hour abs | alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `series_mean_all` | `centered_raw` | 0.0697 | 0.0517 | - | 0.2788 | 0.2051 | - | -0.2765 | - | 0.0091 | 0.9912 | 0.0356 | - |
| `series_mean_all` | `bias_constrained_001` | 0.0697 | 0.0506 | 0.0542 | 0.2788 | 0.2094 | 0.1973 | -0.2086 | -0.0137 | 0.0112 | 0.9961 | 0.0312 | 1.2000 |
| `same_hour_recent_mean_d7_all` | `centered_raw` | 0.0580 | 0.0571 | - | 0.2534 | 0.2427 | - | -0.1262 | - | 0.0001 | -0.8840 | 0.0047 | - |
| `same_hour_recent_mean_d7_all` | `bias_constrained_001` | 0.0580 | 0.0559 | 0.0579 | 0.2534 | 0.2318 | 0.2387 | -0.1372 | 0.0000 | 0.0003 | -0.8946 | 0.0045 | 0.6000 |

## 結果の読み方

`series_mean_all` では、残差補正が明確に効いた。

- `centered_raw`: MAE は 0.0697 から 0.0517 に改善した。
- `bias_constrained_001`: raw corrected MAE は 0.0506、calibrated MAE は 0.0542。
- high residual top10 は calibrated 後に 0.2788 から 0.1973 まで改善した。
- calibrated bias は -0.0137 まで小さくなった。

hour component も強い。

- `centered_raw` の `hour_component_residual_profile_corr` は 0.9912。
- `bias_constrained_001` の `hour_component_residual_profile_corr` は 0.9961。
- `without_hour` delta は 0.0091 から 0.0112 で、hour 成分を消すと MAE が悪化する。

したがって、`series_mean_all` では「残差に hour 構造が残り、提案モデルがそれを拾って補正に使っている」と解釈できる。

一方、`same_hour_recent_mean_d7_all` では改善の性質が違う。

- raw corrected MAE は 0.0580 から 0.0571 / 0.0559 に改善するが、改善幅は小さい。
- calibrated MAE は 0.0579 で baseline とほぼ同水準。
- `without_hour` delta は 0.0001 から 0.0003 と非常に小さい。
- `hour_component_cell_abs_mean` も 0.0045 程度で、`series_mean_all` の 0.031 から 0.036 よりかなり小さい。

これは、`same_hour_recent_mean` baseline が hour 構造を先に吸収してしまい、残差側には hour 成分がほとんど残らない、というこれまでの解釈と一致する。

## 添付結果だけで足りるか

今回の添付 JSON は、metrics の記載には足りる。

具体的には、次は十分に書ける。

- `series_mean_all` と `same_hour_recent_mean_d7_all` の性能差
- hour component の寄与差
- calibration による bias 低減
- high residual top10 の改善

ただし、2-Exp-21 の本来の目的である「図として見せる qualitative analysis」には、標準出力だけでは足りない。

追加で必要なファイルは、実行環境の以下にあるはず。

```text
runs/2-Exp-21_freshretailnet_visualization/
```

特に必要なのは次。

```text
visualization/profiles_by_hour.csv
visualization/profiles_by_day.csv
visualization/series_summary.csv
visualization/series_best_01.csv
visualization/series_worst_01.csv
```

これらがあれば、論文用の hour profile 図、成功例 heatmap、失敗例 heatmap まで作れる。

## ヒートマップ作成

`visualization/` 配下の CSV から SVG を生成するスクリプトを追加した。

```bash
uv run python scripts/plot_2_exp_21_heatmaps.py --root runs/2-Exp-21_freshretailnet_visualization
```

このスクリプトは、各 `visualization/` ディレクトリの下に `figures/` を作る。

出力例:

```text
visualization/figures/profiles_by_hour.svg
visualization/figures/series_best_01_residual.svg
visualization/figures/series_best_01_residual_hat.svg
visualization/figures/series_best_01_hour_component.svg
visualization/figures/series_best_01_baseline_abs_error.svg
visualization/figures/series_best_01_corrected_abs_error.svg
visualization/figures/index.html
```

見る順番は次がよい。

1. `profiles_by_hour.svg` で residual と hour component の対応を見る。
2. `series_best_01_residual.svg` と `series_best_01_residual_hat.svg` を比べる。
3. `series_best_01_baseline_abs_error.svg` と `series_best_01_corrected_abs_error.svg` を比べる。
4. `series_worst_01_*.svg` で、補正が悪化する条件を見る。

## 現時点の結論

2-Exp-21 の数値結果は、2-Exp-19/20 の主張を補強している。

本文では次のように書ける。

```text
series_mean residual では hour component が残差 profile と強く対応し、
hour component を消すと再構成誤差が悪化する。
一方、same-hour recent mean residual では hour component の大きさと寄与が小さく、
強い baseline が hour 構造を吸収していることが示唆される。
```

ただし、図を完成させるには `visualization/` 配下の CSV を確認する必要がある。
