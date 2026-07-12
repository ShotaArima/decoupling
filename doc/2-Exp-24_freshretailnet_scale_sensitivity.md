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
| `centered_raw` | 補正前 calibration なしの4変数への分解 |
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

## 結果

実行コマンド:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-24_freshretailnet_scale_sensitivity.json
```

結果は 3 seed 平均で読む。

| scenario | model | baseline MAE | corrected MAE | calibrated MAE | top10 baseline | top10 corrected | top10 calibrated | bias | calibrated bias | hour corr | hour ablation delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `series_mean_2k` | `mae_grid_reference` | 0.0721 | 0.0598 | 0.0580 | 0.2923 | 0.2688 | 0.2677 | -0.3139 | -0.3575 | 0.9429 | 0.0050 |
| `series_mean_2k` | `bias_constrained_001` | 0.0721 | 0.0598 | 0.0622 | 0.2923 | 0.2643 | 0.2477 | -0.2862 | -0.0002 | 0.9831 | 0.0045 |
| `series_mean_6k` | `mae_grid_reference` | 0.0697 | 0.0510 | 0.0511 | 0.2788 | 0.2132 | 0.2135 | -0.2511 | -0.2825 | 0.9898 | 0.0099 |
| `series_mean_6k` | `bias_constrained_001` | 0.0697 | 0.0514 | 0.0543 | 0.2788 | 0.2126 | 0.1987 | -0.2219 | -0.0123 | 0.9940 | 0.0103 |
| `series_mean_12k` | `mae_grid_reference` | 0.0694 | 0.0494 | 0.0493 | 0.2770 | 0.1994 | 0.2025 | -0.2634 | -0.2717 | 0.9822 | 0.0109 |
| `series_mean_12k` | `bias_constrained_001` | 0.0694 | 0.0487 | 0.0519 | 0.2770 | 0.1964 | 0.1826 | -0.2488 | -0.0139 | 0.9920 | 0.0115 |
| `same_hour_recent_mean_d7_6k` | `mae_grid_reference` | 0.0580 | 0.0564 | 0.0561 | 0.2534 | 0.2384 | 0.2399 | -0.1260 | -0.1628 | -0.8767 | 0.0002 |
| `same_hour_recent_mean_d7_6k` | `bias_constrained_001` | 0.0580 | 0.0565 | 0.0582 | 0.2534 | 0.2401 | 0.2454 | -0.1361 | -0.0000 | -0.8643 | 0.0004 |
| `same_hour_recent_mean_d7_12k` | `mae_grid_reference` | 0.0574 | 0.0546 | 0.0546 | 0.2530 | 0.2322 | 0.2320 | -0.2128 | -0.1917 | -0.8861 | 0.0010 |
| `same_hour_recent_mean_d7_12k` | `bias_constrained_001` | 0.0574 | 0.0547 | 0.0571 | 0.2530 | 0.2307 | 0.2343 | -0.1688 | -0.0002 | -0.8738 | 0.0006 |

## 読み取り

`series_mean` residual では、系列数を 2000 から 12000 に増やしても改善が保たれた。

特に `series_mean_12k` では、baseline MAE 0.0694 に対して corrected MAE は 0.0487〜0.0494 であり、約 29% の改善である。高残差上位 10% でも 0.2770 から 0.1964〜0.1994 まで改善した。bias 制約つき calibration では calibrated top10 が 0.1826 まで下がり、外れケース補正としてはさらに強い。

hour component も安定している。`series_mean_12k` の hour corr は 0.9822〜0.9920 であり、hour component を消したときの MAE 悪化も 0.0109〜0.0115 で、2k / 6k より大きい。系列数を増やすほど、時間帯成分の寄与がむしろ明確になっている。

一方、`same_hour_recent_mean_d7` residual は MAE では少し改善するが、hour corr が一貫して負になっている。これは「予測補正としては多少効くが、hour component が残差の時間帯 profile を素直に説明している」とは言いづらい。強い基準値が時間帯構造を先に吸収し、残った残差では成分の解釈が不安定になる、という限界例として扱うのが妥当である。

## 結論

2-Exp-24 により、FreshRetailNet の主成功例を `series_mean` residual に置く判断は強くなった。

ただし、これはまだ「先頭から取った系列数を増やした」確認であり、FreshRetailNet 全体に対する交差検証ではない。次に必要なのは、系列の取り方を変えても同じ傾向が保たれるかを見ることである。

そのため、次は `2-Exp-25: FreshRetailNet block robustness` を行う。
