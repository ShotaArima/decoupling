# 2-Exp-24: FreshRetailNet Scale Sensitivity

## 背景

2-Exp-23 までで、`series_mean_all` では FreshRetailNet-50K でも残差補正が効くことが分かった。

一方で、これまでの主結果は FreshRetailNet-50K 全体に対する交差検証ではない。

主な FreshRetailNet 実験では、公式 `train` / `eval` split を使い、最大 6000 train 系列、最大 1500 eval 系列を対象にした。

そのため、次に確認すべきことは、系列数を増やしたときに同じ傾向が保たれるかである。

## 目的

目的は 3 つ。

1. `series_mean` residual の改善が系列数を増やしても保たれるか確認する。
2. `same_hour_recent_mean` residual では改善が小さい、という限界解釈が系列数を増やしても保たれるか確認する。
3. hour component の寄与と hour profile corr が、より大きな系列数でも崩れないか確認する。

## 比較条件

| scenario | train 系列上限 | eval 系列上限 | baseline |
|---|---:|---:|---|
| `series_mean_2k` | 2000 | 500 | `series_mean` |
| `series_mean_6k` | 6000 | 1500 | `series_mean` |
| `series_mean_12k` | 12000 | 3000 | `series_mean` |
| `same_hour_recent_mean_d7_6k` | 6000 | 1500 | `same_hour_recent_mean`, recent_days=7 |
| `same_hour_recent_mean_d7_12k` | 12000 | 3000 | `same_hour_recent_mean`, recent_days=7 |

seed は 3 種類にする。

```text
17, 23, 31
```

これは交差検証ではなく、同じ公式 split のもとで、初期値と学習のばらつきを見るための反復である。

## 比較 model

| model | 目的 |
|---|---|
| `centered_raw` | 補正前 calibration なしの出力分解 |
| `mae_grid_reference` | validation MAE を重視した calibration |
| `bias_constrained_001` | validation bias を抑えつつ補正するモデル |

## 成功条件

良い結果:

- `series_mean_12k` でも corrected MAE が baseline MAE より低い。
- `series_mean_12k` でも high residual top10 が改善する。
- `series_mean_12k` でも hour component residual profile corr が高い。
- `same_hour_recent_mean_d7_12k` では改善が小さく、強い baseline が残差構造を吸収するという解釈が保たれる。

悪い結果:

- `series_mean_12k` で改善が消える。
- 系列数を増やすと hour component の対応が崩れる。
- `same_hour_recent_mean_d7_12k` で急に大きく改善し、これまでの限界解釈が不安定になる。

## 実行コマンド

smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-24_freshretailnet_scale_sensitivity_smoke.json
```

本実験:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-24_freshretailnet_scale_sensitivity.json
```

## 出力

```text
runs/2-Exp-24_freshretailnet_scale_sensitivity/all_results.csv
runs/2-Exp-24_freshretailnet_scale_sensitivity/aggregate.csv
runs/2-Exp-24_freshretailnet_scale_sensitivity/summary.json
```

## 論文での使い方

2-Exp-24 は本文の主実験というより、limitation に対する補強である。

結果が良ければ、次のように書ける。

```text
主結果は FreshRetailNet-50K 全体の交差検証ではないが、
系列数を増やした感度確認でも series_mean residual の傾向は保たれた。
```

結果が悪ければ、次のように書く。

```text
series_mean residual の改善は、系列数や対象系列に敏感であり、
提案法の適用条件としてさらに整理が必要である。
```
