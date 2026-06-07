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
