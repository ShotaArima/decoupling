# 2-Exp-19: Bias-Constrained Calibration for FreshRetailNet Residual Correction

## 背景

2-Exp-18 では、FreshRetailNet の `series_mean` 残差に対して残差補正が明確に効いた。

ただし、calibration にはトレードオフがあった。

- MAE を最小にすると、bias が残る。
- bias を消すと、MAE が悪化する。
- high residual top10 は、全体 MAE とは違う動きをする。

このため、次は「bias を完全に 0 にする」ではなく、「許容できる bias の範囲で MAE を最小化する」実験を行う。

## 目的

2-Exp-19 の目的は、残差補正を予測に使うための後処理条件を決めること。

見る点は次の 3 つ。

1. `series_mean_all` で、baseline より低い MAE を維持できるか。
2. raw / MAE grid より corrected bias を小さくできるか。
3. affine calibration より MAE 悪化を抑えられるか。

## 追加した calibration

### `validation_bias_constrained_mae_grid`

validation split で複数の `alpha` を試す。

```text
r_hat_cal = alpha * r_hat + beta
```

各 `alpha` に対して平均 bias が 0 に近くなるように `beta` を推定し、validation 上の予測 bias を測る。

```text
bias_valid = mean(r_hat_cal - r)
```

そして、次を満たす候補だけを残す。

```text
abs(bias_valid) <= max_abs_validation_bias
```

残った候補の中で validation MAE が最小のものを選ぶ。

候補がなければ、bias が最も小さい候補を fallback として使う。

### `validation_weighted_mae_bias_grid`

validation 上で次のスコアを最小化する。

```text
score = MAE + lambda * abs(bias)
```

これは hard constraint ではなく、MAE と bias の重み付き妥協点を見るために使う。

## 比較条件

| scenario | 目的 |
| --- | --- |
| `series_mean_all` | FreshRetailNet の主成功 target |
| `store_third_category_series_mean_repro_top` | 集約粒度でも同じ傾向が出るか |
| `same_hour_recent_mean_d7_all` | 強い baseline では効果が弱まるか |

## 比較モデル

| model | 目的 |
| --- | --- |
| `centered_raw` | calibration なしの基準 |
| `mae_grid_reference` | MAE 優先 calibration |
| `affine_reference` | bias 低減優先 calibration |
| `bias_constrained_001` | validation bias 0.01 以下を狙う |
| `bias_constrained_0025` | validation bias 0.025 以下を狙う |
| `weighted_bias_05` | MAE と bias の重み付き妥協点 |

## 良い結果

- `series_mean_all` で `calibrated_corrected_cell_mae_mean` が baseline より明確に低い。
- `bias_constrained_001` または `bias_constrained_0025` が、`affine_reference` より MAE を保つ。
- `bias_constrained_001` または `bias_constrained_0025` が、`mae_grid_reference` より bias を小さくする。
- high residual top10 でも baseline より改善する。
- `calibrated_calibration_constraint_satisfied_mean` が 1 に近い。

## 悪い結果

- bias 制約を満たすと MAE が affine と同じくらい悪化する。
- 制約を満たす候補が少なく、`constraint_satisfied` が低い。
- `alpha` が 0 近くになり、モデル補正ではなく平均補正になってしまう。
- high residual top10 の改善が消える。

## 実行コマンド

まず smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-19_smoke.json
```

本実験:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-19_freshretailnet_bias_constrained_calibration.json
```

## 見る指標

- `calibrated_corrected_cell_mae_mean`
- `calibrated_corrected_cell_bias_mean`
- `calibrated_high_residual_top10_corrected_mae_mean`
- `calibrated_calibration_alpha_mean`
- `calibrated_calibration_validation_prediction_bias_mean`
- `calibrated_calibration_constraint_satisfied_mean`
- `component_ablation_without_hour_mae_delta_mean`
- `hour_component_residual_profile_corr_mean`

## 論文上の位置づけ

2-Exp-19 は、FreshRetailNet を「実データでの限界検証」から一歩進めて、「実際に補正器として使う場合の条件」を調べる実験である。

この実験で中間解が見つかれば、論文では次のように書ける。

1. 残差分解は hour 成分を安定して拾う。
2. raw 補正は MAE を下げるが bias を持つ。
3. validation calibration により bias と MAE の trade-off を制御できる。
4. したがって、本手法は baseline を置き換えるのではなく、baseline の残差補正器として使うのが自然である。

## 実行結果

5 seed の本実験は完走した。

| scenario | model | baseline MAE | corrected MAE | calibrated MAE | corrected bias | calibrated bias | valid bias | constraint | top10 baseline | top10 calibrated | alpha | hour delta | hour corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `series_mean_all` | `centered_raw` | 0.0697 | 0.0506 | - | -0.2567 | - | - | - | 0.2788 | - | - | 0.0099 | 0.9934 |
| `series_mean_all` | `mae_grid_reference` | 0.0697 | 0.0508 | 0.0509 | -0.2535 | -0.2835 | -0.0161 | - | 0.2788 | 0.2127 | 1.0400 | 0.0101 | 0.9897 |
| `series_mean_all` | `affine_reference` | 0.0697 | 0.0508 | 0.0553 | -0.2380 | -0.0093 | -0.0000 | - | 0.2788 | 0.2043 | 1.0164 | 0.0102 | 0.9924 |
| `series_mean_all` | `bias_constrained_001` | 0.0697 | 0.0501 | 0.0534 | -0.2973 | -0.0141 | -0.0000 | 1.0000 | 0.2788 | 0.1914 | 1.2000 | 0.0097 | 0.9938 |
| `series_mean_all` | `bias_constrained_0025` | 0.0697 | 0.0512 | 0.0543 | -0.2295 | -0.0122 | -0.0000 | 1.0000 | 0.2788 | 0.1964 | 1.2000 | 0.0100 | 0.9958 |
| `series_mean_all` | `weighted_bias_05` | 0.0697 | 0.0516 | 0.0553 | -0.2979 | -0.0122 | 0.0000 | - | 0.2788 | 0.2014 | 1.2000 | 0.0089 | 0.9819 |
| `store_third_category_series_mean_repro_top` | `centered_raw` | 0.0811 | 0.0643 | - | -0.2369 | - | - | - | 0.3138 | - | - | 0.0096 | 0.9857 |
| `store_third_category_series_mean_repro_top` | `mae_grid_reference` | 0.0811 | 0.0637 | 0.0632 | -0.3026 | -0.2693 | -0.0187 | - | 0.3138 | 0.2440 | 1.0200 | 0.0099 | 0.9906 |
| `store_third_category_series_mean_repro_top` | `affine_reference` | 0.0811 | 0.0648 | 0.0658 | -0.2544 | -0.0001 | 0.0000 | - | 0.3138 | 0.2190 | 1.4814 | 0.0099 | 0.9907 |
| `store_third_category_series_mean_repro_top` | `bias_constrained_001` | 0.0811 | 0.0643 | 0.0667 | -0.2512 | -0.0000 | -0.0000 | 1.0000 | 0.3138 | 0.2372 | 1.1800 | 0.0092 | 0.9885 |
| `store_third_category_series_mean_repro_top` | `bias_constrained_0025` | 0.0811 | 0.0639 | 0.0665 | -0.2651 | -0.0000 | 0.0000 | 1.0000 | 0.3138 | 0.2363 | 1.1800 | 0.0094 | 0.9924 |
| `store_third_category_series_mean_repro_top` | `weighted_bias_05` | 0.0811 | 0.0637 | 0.0654 | -0.2723 | -0.0001 | 0.0000 | - | 0.3138 | 0.2211 | 1.1800 | 0.0098 | 0.9882 |
| `same_hour_recent_mean_d7_all` | `centered_raw` | 0.0580 | 0.0561 | - | -0.1612 | - | - | - | 0.2534 | - | - | 0.0006 | -0.8772 |
| `same_hour_recent_mean_d7_all` | `mae_grid_reference` | 0.0580 | 0.0564 | 0.0561 | -0.1429 | -0.1683 | -0.0099 | - | 0.2534 | 0.2404 | 1.0800 | 0.0004 | -0.8684 |
| `same_hour_recent_mean_d7_all` | `affine_reference` | 0.0580 | 0.0564 | 0.0595 | -0.1674 | -0.0000 | -0.0000 | - | 0.2534 | 0.2338 | 1.4693 | 0.0007 | -0.8622 |
| `same_hour_recent_mean_d7_all` | `bias_constrained_001` | 0.0580 | 0.0565 | 0.0584 | -0.1692 | -0.0000 | -0.0000 | 1.0000 | 0.2534 | 0.2480 | 0.3400 | 0.0008 | -0.8766 |
| `same_hour_recent_mean_d7_all` | `bias_constrained_0025` | 0.0580 | 0.0567 | 0.0584 | -0.1564 | -0.0000 | 0.0000 | 1.0000 | 0.2534 | 0.2497 | 0.2400 | 0.0005 | -0.8521 |
| `same_hour_recent_mean_d7_all` | `weighted_bias_05` | 0.0580 | 0.0561 | 0.0583 | -0.1830 | -0.0000 | 0.0000 | - | 0.2534 | 0.2457 | 0.4000 | 0.0008 | -0.8921 |

## 結果の読み方

`series_mean_all` では、`bias_constrained_001` が calibrated MAE 0.0534 で baseline 0.0697 を大きく下回った。さらに calibrated bias は -0.0141 まで縮み、high residual top10 は 0.1914 で最も良い。これは「全体 MAE だけを最小にするモデル」ではなく、「外れケースと bias を重視する補正器」として有望である。

`store_third_category_series_mean_repro_top` では、`mae_grid_reference` が calibrated MAE 0.0632 で最良だった。bias 制約モデルは bias をほぼ 0 にできるが、MAE は 0.066 台まで悪化する。ただし baseline 0.0811 よりは十分良い。

`same_hour_recent_mean_d7_all` では、強い baseline のため calibrated MAE の改善は弱い。bias 制約モデルは MAE では baseline を少し上回ってしまうが、high residual top10 は baseline より良い。この条件は主成功ではなく、強い baseline 下での限界例として扱う。

## 次の判断

2-Exp-19 だけで平均値の方向性は見えた。

ただし論文表にするには、5 seed の paired 差分に対して信頼区間を出す必要がある。次は `2-Exp-20` で seed-level paired bootstrap を行う。
