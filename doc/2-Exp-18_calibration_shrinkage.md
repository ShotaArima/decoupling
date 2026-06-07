# 2-Exp-18: Calibration and Shrinkage for FreshRetailNet Residual Correction

## 背景

2-Exp-17 では、`series_mean_all` と `store_third_category_series_mean_repro_top` が 5 seed で安定して改善した。

ただし、raw の `r_hat` には負方向の bias が残った。

- `series_mean_all / output_decomp_centered`: corrected bias は約 -0.257
- `series_mean_all / output_decomp_bias_loss_calibrated`: corrected bias は約 -0.182
- `store_third_category_series_mean_repro_top / output_decomp_centered`: corrected bias は約 -0.237

一方で、hour 成分の寄与は明確だった。

- `series_mean_all / output_decomp_centered`: `without_hour` delta は約 0.0099
- `series_mean_all / output_decomp_bias_loss_calibrated`: `without_hour` delta は約 0.0115
- `hour_component_residual_profile_corr` は `series_mean_all` で約 0.99
- `store_third_category_series_mean_repro_top` でも約 0.99

つまり、hour profile は正しく拾えているが、補正量の水準がずれている。

## 目的

2-Exp-18 では、hour 成分を保ったまま `r_hat` の大きさと bias を validation split で補正する。

これはモデル構造の主張ではなく、実運用上の予測補正として必要な後処理である。

## 追加した calibration

### `validation_affine_residual`

validation 上で次を最小二乗で合わせる。

```text
r = alpha * r_hat + beta
```

その後、test では

```text
r_hat_cal = alpha * r_hat + beta
y_hat = b + r_hat_cal
```

を見る。

良い結果:

- bias が縮む
- MAE が raw と同等、または改善する
- `alpha` が 0 ではなく、hour 成分の情報を残している

悪い結果:

- `alpha` が 0 近くになり、単なる平均補正になる
- bias は縮むが MAE が悪化する

### `validation_mae_grid`

validation 上で `alpha` を grid search し、各 `alpha` に対して median bias を入れる。

```text
beta(alpha) = median(r - alpha * r_hat)
```

そして validation MAE が最小の `alpha` を選ぶ。

これは外れ値に強い shrinkage として使う。

良い結果:

- `series_mean_all` と `store_third_category_series_mean_repro_top` で raw の改善を保ったまま bias が縮む
- high residual top10 でも baseline より改善する
- `alpha` が 0.3 から 1.0 程度に残り、モデル補正が使われている

悪い結果:

- `alpha = 0` が選ばれる
- high residual top10 の改善が消える
- `same_hour_recent_mean_d7_all` でだけ良く、主成功 target では効かない

## 比較条件

| scenario | 役割 |
| --- | --- |
| `series_mean_all` | FreshRetailNet の主成功 target |
| `store_third_category_series_mean_repro_top` | 集約粒度での主成功 target |
| `same_hour_recent_mean_d7_all` | 強い baseline 対照 |

## 比較モデル

| model | 目的 |
| --- | --- |
| `centered_raw` | 2-Exp-17 の基本モデル |
| `centered_bias_calibrated` | 既存の validation bias 補正 |
| `centered_affine_calibrated` | validation affine 補正 |
| `centered_mae_grid_calibrated` | validation MAE 最小 shrinkage |
| `bias_loss_mae_grid_calibrated` | bias loss 付きモデルに MAE shrinkage を適用 |

## 成功条件

主成功:

- `series_mean_all` で calibrated corrected MAE が baseline MAE より低い
- `series_mean_all` で calibrated high residual top10 が baseline より低い
- calibrated corrected bias の絶対値が raw より小さい
- `calibrated_calibration_alpha_mean` が 0 より十分大きい

補助成功:

- `store_third_category_series_mean_repro_top` でも同じ傾向が出る
- `same_hour_recent_mean_d7_all` では改善が小さく、強い baseline の限界を示せる
- hour profile 相関は高いまま維持される

失敗条件:

- shrinkage 後に MAE 改善が消える
- `alpha` が 0 になり、モデル補正が使われない
- bias は消えるが high residual top10 が悪化する

## 実行コマンド

まず smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-18_smoke.json
```

本実験:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-18_freshretailnet_calibration_shrinkage.json
```

本実験 config では、ディスク容量を抑えるために大きい出力を保存しない。

```json
"output": {
  "save_latent_arrays": false,
  "save_residual_predictions": false,
  "save_checkpoints": false,
  "save_hour_heatmap": true
}
```

これにより、`metrics.json`、`summary.json`、`aggregate.csv`、`all_results.csv`、小さい `z_hour_heatmap.csv` は残しつつ、容量を消費しやすい `*.npy`、cell-level prediction、checkpoint は保存しない。

## 見る指標

- `corrected_cell_mae_mean`
- `calibrated_corrected_cell_mae_mean`
- `corrected_cell_bias_mean`
- `calibrated_corrected_cell_bias_mean`
- `high_residual_top10_corrected_mae_mean`
- `calibrated_high_residual_top10_corrected_mae_mean`
- `calibrated_calibration_alpha_mean`
- `calibrated_calibration_residual_bias_mean`
- `calibrated_calibration_validation_mae_mean`
- `hour_component_residual_profile_corr_mean`
- `calibrated_hour_component_residual_profile_corr_mean`

## 論文上の位置づけ

2-Exp-17 で「hour 成分は実データでも残差補正に効く」ことを示した。

2-Exp-18 では「その補正を実際の予測に使うには validation calibration が必要」という実務的な条件を示す。

この結果が良ければ、FreshRetailNet の節では次の流れで書ける。

1. 強い baseline では残差構造が消える。
2. `series_mean` target では hour 構造が残る。
3. output decomposition は hour 構造を拾う。
4. raw 補正は bias を持つ。
5. validation shrinkage で bias を抑えると、予測補正としても使いやすくなる。
