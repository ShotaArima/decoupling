# EXP-004〜EXP-007: 次段階の実験設計

## 目的

`EXP-003` では、多粒度 local 分離仮説を検証するための実験基盤を作りました。次段階では、以下を順番に確認します。

1. `valid WAPE` による checkpoint 選択と early stopping が正しく機能するか。
2. synthetic / FreshRetailNet で 30 epoch の本実験を行い、naive baseline と提案モデルを比較する。
3. naive baseline と proposed model が同じ test set・同じ評価指標・同じ出力形式で比較できているか確認する。
4. 欠品重みあり/なし、`z_day` probe、`z_hour` heatmap を使い、予測性能だけでなく解釈性を評価する。

## 追加実装

`src/decoupled_ts/retail_experiments.py` に以下を追加しました。

| 機能 | 内容 |
|---|---|
| valid prediction metrics | 各 epoch で validation set の `valid_mae`, `valid_rmse`, `valid_wape`, `valid_bias` を計算 |
| valid WAPE checkpoint | `train.selection_metric = "valid_wape"` で best checkpoint を選択 |
| early stopping | `early_stopping_patience` と `early_stopping_min_delta` を追加 |
| stockout weighting | auxiliary reconstruction loss で `mask`, `soft`, `uniform` を比較可能 |
| z_day probe | `z_day` から weekday / holiday / discount を予測する probe を追加 |
| z_hour heatmap | `z_hour_heatmap.csv` を variant ごとに出力 |
| latent diagnostics | `latent_diagnostics.json` に day/hour probe 補助情報を保存 |

## EXP-004: valid WAPE checkpoint と early stopping

### 問い

```text
total loss ではなく、需要予測の主指標である valid WAPE で checkpoint を選ぶと、
test WAPE / bias の評価が安定するか？
```

### 実行コマンド

動作確認:

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-004_valid_wape_checkpoint_smoke.json
```

FreshRetailNet:

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-004_valid_wape_checkpoint_freshretailnet.json
```

### 結果取得先

| 実験 | 主な結果ファイル |
|---|---|
| smoke | `runs/EXP-004_valid_wape_checkpoint_smoke/summary.csv` |
| smoke | `runs/EXP-004_valid_wape_checkpoint_smoke/summary.json` |
| smoke | `runs/EXP-004_valid_wape_checkpoint_smoke/*/history.jsonl` |
| FreshRetailNet | `runs/EXP-004_valid_wape_checkpoint_freshretailnet/summary.csv` |
| FreshRetailNet | `runs/EXP-004_valid_wape_checkpoint_freshretailnet/summary.json` |
| FreshRetailNet | `runs/EXP-004_valid_wape_checkpoint_freshretailnet/*/history.jsonl` |

確認すべき列:

```text
selection_metric
best_validation_score
best_epoch
wape
mae
rmse
bias
```

`history.jsonl` では以下を見ます。

```text
valid_wape
valid_mae
valid_rmse
valid_bias
valid_loss_forecast
```

## EXP-005: synthetic / FreshRetailNet で 30 epoch 本実験

### 問い

```text
1 epoch の smoke ではなく、30 epoch + early stopping で、
提案モデルは naive baseline / feature MLP / global-only と比較して有利か？
```

### 実行コマンド

Synthetic:

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-005_30epoch_synthetic.json
```

FreshRetailNet:

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-005_30epoch_freshretailnet.json
```

### 比較対象

| model | 目的 |
|---|---|
| `naive_last_day` | 直近日の単純コピー |
| `naive_recent_mean` | 直近平均 baseline |
| `naive_same_hour_recent_mean` | 時間帯別の直近平均 baseline |
| `feature_flatten_mlp` | 単純特徴量追加 MLP |
| `global_only` | 系列固有成分のみ |
| `global_day` | day 成分の寄与 |
| `global_hour` | hour 成分の寄与 |
| `proposed_no_decouple` | 構造分離のみ |
| `proposed_with_decouple` | 構造分離 + decouple loss |
| `proposed_interaction` | day-hour interaction 追加 |

### 結果取得先

| 実験 | 主な結果ファイル |
|---|---|
| Synthetic | `runs/EXP-005_30epoch_synthetic/summary.csv` |
| Synthetic | `runs/EXP-005_30epoch_synthetic/summary.json` |
| Synthetic | `runs/EXP-005_30epoch_synthetic/data_audit.json` |
| Synthetic | `runs/EXP-005_30epoch_synthetic/*/metrics.json` |
| Synthetic | `runs/EXP-005_30epoch_synthetic/*/predictions.csv` |
| FreshRetailNet | `runs/EXP-005_30epoch_freshretailnet/summary.csv` |
| FreshRetailNet | `runs/EXP-005_30epoch_freshretailnet/summary.json` |
| FreshRetailNet | `runs/EXP-005_30epoch_freshretailnet/data_audit.json` |
| FreshRetailNet | `runs/EXP-005_30epoch_freshretailnet/*/metrics.json` |
| FreshRetailNet | `runs/EXP-005_30epoch_freshretailnet/*/predictions.csv` |

最初に見るべきファイルは `summary.csv` です。需要帯別の過小予測を見る場合は `predictions.csv` を使います。

## EXP-006: naive baseline と proposed の比較可能性確認

### 問い

```text
naive baseline と proposed model は、同じ test set、同じ予測対象、同じ指標で比較できているか？
```

### 実行コマンド

Synthetic:

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-006_baseline_comparability_synthetic.json
```

FreshRetailNet:

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-006_baseline_comparability_freshretailnet.json
```

### 確認すること

- 全 variant に `predictions.csv` が生成されているか。
- `predictions.csv` の行数が variant 間で一致するか。
- `summary.csv` の WAPE / MAE / RMSE / bias が `predictions.csv` から再計算できるか。
- naive が強い場合、proposed が負けている理由が bias なのか、外れ値なのか、需要帯によるものなのか。

### 結果取得先

| 実験 | 主な結果ファイル |
|---|---|
| Synthetic | `runs/EXP-006_baseline_comparability_synthetic/summary.csv` |
| Synthetic | `runs/EXP-006_baseline_comparability_synthetic/*/predictions.csv` |
| Synthetic | `runs/EXP-006_baseline_comparability_synthetic/data_audit.json` |
| FreshRetailNet | `runs/EXP-006_baseline_comparability_freshretailnet/summary.csv` |
| FreshRetailNet | `runs/EXP-006_baseline_comparability_freshretailnet/*/predictions.csv` |
| FreshRetailNet | `runs/EXP-006_baseline_comparability_freshretailnet/data_audit.json` |

## EXP-007: 欠品重み比較と probe / heatmap

### 問い

```text
欠品時間帯の扱いを変えると、予測性能・bias・潜在表現の解釈性は変わるか？
また、z_day と z_hour は想定した情報を持っているか？
```

### 実行コマンド

Synthetic:

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-007_stockout_probe_heatmap_synthetic.json
```

FreshRetailNet:

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-007_stockout_probe_heatmap_freshretailnet.json
```

### 比較する欠品重み

| variant | `stockout_weighting` | 意味 |
|---|---|---|
| `proposed_stockout_mask` | `mask` | 欠品中の sales reconstruction を無視 |
| `proposed_stockout_soft_0_1` | `soft` | 欠品中も 0.1 の重みで補助損失に入れる |
| `proposed_stockout_uniform` | `uniform` | 欠品中も通常観測と同じ重みで扱う |

### 結果取得先

| 実験 | 主な結果ファイル |
|---|---|
| Synthetic | `runs/EXP-007_stockout_probe_heatmap_synthetic/summary.csv` |
| Synthetic | `runs/EXP-007_stockout_probe_heatmap_synthetic/*/metrics.json` |
| Synthetic | `runs/EXP-007_stockout_probe_heatmap_synthetic/*/z_hour_heatmap.csv` |
| Synthetic | `runs/EXP-007_stockout_probe_heatmap_synthetic/*/latent_diagnostics.json` |
| Synthetic | `runs/EXP-007_stockout_probe_heatmap_synthetic/*/z_day.npy` |
| Synthetic | `runs/EXP-007_stockout_probe_heatmap_synthetic/*/z_hour.npy` |
| FreshRetailNet | `runs/EXP-007_stockout_probe_heatmap_freshretailnet/summary.csv` |
| FreshRetailNet | `runs/EXP-007_stockout_probe_heatmap_freshretailnet/*/metrics.json` |
| FreshRetailNet | `runs/EXP-007_stockout_probe_heatmap_freshretailnet/*/z_hour_heatmap.csv` |
| FreshRetailNet | `runs/EXP-007_stockout_probe_heatmap_freshretailnet/*/latent_diagnostics.json` |
| FreshRetailNet | `runs/EXP-007_stockout_probe_heatmap_freshretailnet/*/z_day.npy` |
| FreshRetailNet | `runs/EXP-007_stockout_probe_heatmap_freshretailnet/*/z_hour.npy` |

### probe で見る指標

`summary.csv` / `metrics.json` に以下が出ます。

```text
probe_z_global_subgroup_accuracy
probe_subgroup_majority_accuracy
probe_subgroup_train_classes
probe_subgroup_test_classes
probe_subgroup_overlap_classes
probe_z_day_weekday_accuracy
probe_z_day_holiday_accuracy
probe_z_day_discount_mae
```

`z_hour_heatmap.csv` は、行が hour、列が latent dimension です。時間帯ごとの潜在表現の平均パターンを見るために使います。

## 実行順序の推奨

まず smoke でコード確認します。

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-004_valid_wape_checkpoint_smoke.json
```

次に本実験です。

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-005_30epoch_synthetic.json
uv run decoupled-ts retail-experiment --config configs/EXP-005_30epoch_freshretailnet.json
```

その結果を見て、naive が強い場合は EXP-006 を確認します。

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-006_baseline_comparability_synthetic.json
uv run decoupled-ts retail-experiment --config configs/EXP-006_baseline_comparability_freshretailnet.json
```

最後に、表現解釈と欠品重みを評価します。

```bash
uv run decoupled-ts retail-experiment --config configs/EXP-007_stockout_probe_heatmap_synthetic.json
uv run decoupled-ts retail-experiment --config configs/EXP-007_stockout_probe_heatmap_freshretailnet.json
```
