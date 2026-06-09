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

## 実行結果

ディスク容量対策後の本実験は正常に完走した。

5 seed の集計は次の通り。

| scenario | model | baseline MAE | raw/corrected MAE | calibrated MAE | corrected bias | calibrated bias | top10 corrected | top10 calibrated | alpha | hour ablation delta | hour corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `series_mean_all` | `centered_raw` | 0.0697 | 0.0506 | - | -0.2567 | - | 0.2061 | - | - | 0.0099 | 0.9934 |
| `series_mean_all` | `centered_bias_calibrated` | 0.0697 | 0.0508 | 0.0557 | -0.2535 | -0.0101 | 0.2110 | 0.2058 | 1.0000 | 0.0101 | 0.9897 |
| `series_mean_all` | `centered_affine_calibrated` | 0.0697 | 0.0508 | 0.0553 | -0.2380 | -0.0093 | 0.2101 | 0.2043 | 1.0164 | 0.0102 | 0.9924 |
| `series_mean_all` | `centered_mae_grid_calibrated` | 0.0697 | 0.0501 | 0.0501 | -0.2973 | -0.2783 | 0.2071 | 0.2066 | 1.0600 | 0.0097 | 0.9938 |
| `series_mean_all` | `bias_loss_mae_grid_calibrated` | 0.0697 | 0.0511 | 0.0510 | -0.1699 | -0.2487 | 0.1974 | 0.2085 | 0.9000 | 0.0112 | 0.9965 |
| `store_third_category_series_mean_repro_top` | `centered_raw` | 0.0811 | 0.0643 | - | -0.2369 | - | 0.2509 | - | - | 0.0096 | 0.9857 |
| `store_third_category_series_mean_repro_top` | `centered_bias_calibrated` | 0.0811 | 0.0637 | 0.0668 | -0.3026 | -0.0000 | 0.2458 | 0.2371 | 1.0000 | 0.0099 | 0.9906 |
| `store_third_category_series_mean_repro_top` | `centered_affine_calibrated` | 0.0811 | 0.0648 | 0.0658 | -0.2544 | -0.0001 | 0.2554 | 0.2190 | 1.4814 | 0.0099 | 0.9907 |
| `store_third_category_series_mean_repro_top` | `centered_mae_grid_calibrated` | 0.0811 | 0.0643 | 0.0640 | -0.2512 | -0.2749 | 0.2554 | 0.2551 | 1.0400 | 0.0092 | 0.9885 |
| `store_third_category_series_mean_repro_top` | `bias_loss_mae_grid_calibrated` | 0.0811 | 0.0656 | 0.0641 | -0.0527 | -0.2579 | 0.2411 | 0.2540 | 0.9000 | 0.0124 | 0.9868 |
| `same_hour_recent_mean_d7_all` | `centered_raw` | 0.0580 | 0.0561 | - | -0.1612 | - | 0.2403 | - | - | 0.0006 | -0.8772 |
| `same_hour_recent_mean_d7_all` | `centered_bias_calibrated` | 0.0580 | 0.0564 | 0.0586 | -0.1429 | -0.0001 | 0.2406 | 0.2382 | 1.0000 | 0.0004 | -0.8684 |
| `same_hour_recent_mean_d7_all` | `centered_affine_calibrated` | 0.0580 | 0.0564 | 0.0595 | -0.1674 | -0.0000 | 0.2423 | 0.2338 | 1.4693 | 0.0007 | -0.8622 |
| `same_hour_recent_mean_d7_all` | `centered_mae_grid_calibrated` | 0.0580 | 0.0565 | 0.0564 | -0.1692 | -0.1721 | 0.2443 | 0.2439 | 1.0600 | 0.0008 | -0.8766 |
| `same_hour_recent_mean_d7_all` | `bias_loss_mae_grid_calibrated` | 0.0580 | 0.0569 | 0.0568 | -0.1265 | -0.1282 | 0.2473 | 0.2458 | 1.2000 | 0.0005 | -0.8765 |

## 考察

`series_mean_all` は、baseline MAE 0.0697 に対して corrected MAE が 0.050 から 0.051 まで下がった。これは FreshRetailNet でも、残差補正が予測誤差を明確に下げる条件があることを示している。

`store_third_category_series_mean_repro_top` でも、baseline MAE 0.0811 に対して corrected MAE が 0.064 前後まで下がった。カテゴリ集約でも同じ方向の改善が出ているため、店舗商品単位だけの偶然ではない。

一方で、calibration の目的によって結果は分かれた。

- `validation_affine_residual` と `validation_residual_bias` は bias をほぼ 0 にできる。
- ただし MAE は `series_mean_all` で 0.050 台から 0.055 台へ悪化した。
- `validation_mae_grid` は MAE を守るが、bias はかなり残る。
- high residual top10 では affine が改善する場面もあり、全体 MAE と外れケース改善は同じ目的ではない。

hour 成分については、`series_mean_all` とカテゴリ集約で `hour_component_residual_profile_corr` が 0.99 前後、`without_hour` delta も 0.009 から 0.012 程度で安定している。したがって「hour 成分は残差補正に効いている」という主張は強くなった。

`same_hour_recent_mean_d7_all` は baseline が強く、hour 成分の寄与が小さい。これはこれまでの仮説通り、強い baseline の後には学習可能な hour 構造が薄くなることを示している。

## 次の実験

2-Exp-18 の残課題は、MAE と bias のトレードオフである。

次は `2-Exp-19` として、validation 上で「bias が一定以下の候補から MAE 最小」を選ぶ calibration を試す。

目的は、次の中間解を探すこと。

- raw / MAE grid より bias が小さい。
- affine / bias calibration より MAE が悪化しにくい。
- high residual top10 の改善を残す。

この結果が良ければ、FreshRetailNet 節では「表現分解による残差補正は有効だが、実運用では bias 制約つき calibration が必要」と整理できる。
