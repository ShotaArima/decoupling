# 2-Exp-2〜2-Exp-6: 残差表現学習と補正評価

## 目的

`2-Exp-1` で定義した基準成分 `b` を使い、売上そのものではなく

```text
r = y - b
```

を対象にした表現学習を行います。

この runner は、以下をまとめて出力します。

| 実験 | 対応 Step | 実装上の出力 |
|---|---|---|
| `2-Exp-2` | Step 4 | `r` を入力・再構成対象にした latent model の学習 |
| `2-Exp-3` | Step 5 | `residual_flatten_ae`, `residual_global_single_local`, `residual_global_day_hour` などの再構成比較 |
| `2-Exp-4` | Step 5 | `z_global`, `z_day`, `z_hour` probe |
| `2-Exp-5` | Step 5 | `z_hour_heatmap.csv`, latent `.npy` |
| `2-Exp-6` | Step 6 | cell-level の `b` vs `b + r_hat` 補正評価 |

## 実行コマンド

Smoke:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-2_to_6_residual_smoke.json
```

Synthetic:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-2_to_6_residual_synthetic.json
```

FreshRetailNet:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-2_to_6_residual_freshretailnet.json
```

## 比較モデル

| variant | 意味 |
|---|---|
| `residual_flatten_ae` | 分離なしの flatten autoencoder |
| `residual_global_only` | 系列固有成分のみ |
| `residual_global_single_local` | `z_global + z_local` の単一 local |
| `residual_global_day_hour_no_decouple` | `z_global + z_day + z_hour`、分離制約なし |
| `residual_global_day_hour` | `z_global + z_day + z_hour`、分離制約あり |
| `residual_global_day_hour_interaction` | `z_day_hour` interaction 追加 |

## 主指標

残差再構成:

```text
residual_mae
residual_rmse
residual_r2
residual_sign_accuracy
```

下流補正:

```text
baseline_cell_mae
corrected_cell_mae
baseline_cell_rmse
corrected_cell_rmse
baseline_cell_wape
corrected_cell_wape
high_residual_top10_baseline_mae
high_residual_top10_corrected_mae
```

probe:

```text
probe_z_global_subgroup_accuracy
probe_z_day_weekday_accuracy
probe_z_day_discount_mae
probe_z_hour_hour_accuracy
```

## 出力

各 run の root:

| file | 内容 |
|---|---|
| `summary.json` | 全 variant 結果 |
| `summary.csv` | 比較表 |
| `resolved_config.json` | 実行時 config |
| `run.log` | 実行ログ |

各 variant:

| file | 内容 |
|---|---|
| `history.jsonl` | epoch ごとの train/valid loss |
| `metrics.json` | 再構成・補正・probe 指標 |
| `residual_predictions.csv` | cell-level の `sales`, `baseline`, `residual`, `residual_hat`, `corrected` |
| `z_global.npy`, `z_day.npy`, `z_hour.npy` | latent arrays |
| `z_hour_heatmap.csv` | hour ごとの latent 平均 |

## 注意

現状の `2-Exp-6` は、history window 内の observed cell に対する補正評価です。
将来期間の `b + r_hat` 予測補正へ進めるには、future covariate をどう与えるか、または `z_day` の将来推定をどう置くかを別途設計する必要があります。
