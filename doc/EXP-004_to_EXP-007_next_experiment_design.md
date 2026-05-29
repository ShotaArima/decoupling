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

## 実験結果

### EXP-004: valid WAPE checkpoint smoke

| model | best epoch | valid WAPE | test WAPE | MAE | RMSE | bias |
|---|---:|---:|---:|---:|---:|---:|
| `feature_flatten_mlp` | 5 | 0.6691 | 0.6364 | 35.7364 | 42.6594 | -0.6364 |
| `proposed_with_decouple` | 5 | 0.6749 | 0.6057 | 34.0126 | 42.2566 | -0.6054 |

所見:

- `valid_wape` による checkpoint 選択は機能しています。
- smoke では `proposed_with_decouple` が `feature_flatten_mlp` より test WAPE / MAE / RMSE で良いです。
- ただし両方とも強い負の bias があり、smoke は性能評価ではなくコード確認として扱います。

### EXP-005: 30 epoch synthetic

| model | best epoch | valid WAPE | test WAPE | MAE | RMSE | bias |
|---|---:|---:|---:|---:|---:|---:|
| `naive_last_day` | - | - | 0.2760 | 31.4417 | 45.2243 | 0.2099 |
| `naive_recent_mean` | - | - | 0.1479 | 16.8461 | 21.8661 | 0.0873 |
| `naive_same_hour_recent_mean` | - | - | 0.1493 | 17.0031 | 22.2150 | 0.0914 |
| `feature_flatten_mlp` | 11 | 0.1164 | 0.1042 | 11.8706 | 15.9041 | -0.0094 |
| `global_only` | 28 | 0.1136 | **0.0979** | **11.1536** | 15.3251 | -0.0105 |
| `global_day` | 29 | 0.1132 | 0.0996 | 11.3436 | 15.6763 | -0.0199 |
| `global_hour` | 17 | 0.1136 | 0.0998 | 11.3690 | 15.4872 | 0.0005 |
| `proposed_no_decouple` | 12 | 0.1169 | 0.1006 | 11.4550 | 15.3523 | -0.0026 |
| `proposed_with_decouple` | 21 | 0.1139 | 0.0994 | 11.3211 | 15.6162 | -0.0178 |
| `proposed_interaction` | 13 | 0.1175 | 0.1024 | 11.6662 | **15.3131** | 0.0067 |

所見:

- synthetic では学習モデル群が naive baseline を明確に上回りました。
- 最良 WAPE / MAE は `global_only` です。
- `proposed_with_decouple` は `feature_flatten_mlp` より良いですが、`global_only` には届いていません。
- `z_day` probe は一部モデルで weekday / holiday を強く捉えていますが、提案モデルが最良ではありません。
- この synthetic 設定では、day/hour 分離の有効性よりも global 表現の強さが目立ちます。

### EXP-005: 30 epoch FreshRetailNet

| model | best epoch | valid WAPE | test WAPE | MAE | RMSE | bias |
|---|---:|---:|---:|---:|---:|---:|
| `naive_last_day` | - | - | 0.3622 | 0.8629 | 1.4713 | 0.0804 |
| `naive_recent_mean` | - | - | 0.5084 | 1.2112 | 2.2258 | 0.4704 |
| `naive_same_hour_recent_mean` | - | - | **0.3405** | **0.8112** | **1.3154** | 0.2714 |
| `feature_flatten_mlp` | 8 | 0.6161 | 0.6161 | 1.4679 | 4.2142 | -0.2751 |
| `global_only` | 1 | 0.6220 | 0.6220 | 1.4818 | 4.3348 | -0.3758 |
| `global_day` | 5 | 0.6161 | 0.6161 | 1.4679 | 4.3257 | -0.3686 |
| `global_hour` | 2 | 0.6747 | 0.6747 | 1.6075 | 4.4868 | -0.6154 |
| `proposed_no_decouple` | 2 | 0.6166 | 0.6166 | 1.4691 | 4.3291 | -0.3654 |
| `proposed_with_decouple` | 3 | 0.6176 | 0.6176 | 1.4714 | 4.3521 | -0.4147 |
| `proposed_interaction` | 7 | 0.6181 | 0.6181 | 1.4727 | 4.3496 | -0.4195 |

所見:

- FreshRetailNet では `naive_same_hour_recent_mean` が最良です。
- 学習モデル群は naive baseline に大きく負けています。
- 学習モデルは全体的に負の bias が大きく、過小予測に寄っています。
- この結果から、FreshRetailNet では現状の提案モデルを「予測性能で有利」と主張するのは難しいです。
- 次の改善対象は、future day の既知特徴利用、target scale の扱い、店舗×カテゴリ集約、または予測 head の再設計です。

### EXP-006: baseline comparability synthetic

| model | best epoch | valid WAPE | test WAPE | MAE | RMSE | bias |
|---|---:|---:|---:|---:|---:|---:|
| `naive_recent_mean` | - | - | 0.1452 | 19.6095 | 29.1819 | 0.0732 |
| `naive_same_hour_recent_mean` | - | - | 0.1477 | 19.9495 | 29.6868 | 0.0772 |
| `feature_flatten_mlp` | 8 | 0.1037 | **0.1027** | **13.8713** | **25.6710** | -0.0153 |
| `proposed_with_decouple` | 17 | **0.1008** | 0.1030 | 13.9091 | 25.8636 | -0.0199 |

所見:

- synthetic では learned model が naive baseline を上回っています。
- `feature_flatten_mlp` と `proposed_with_decouple` はほぼ同等です。
- valid WAPE は proposed が良い一方、test WAPE は feature MLP がわずかに良いです。
- 予測性能だけでは、提案モデルの優位性は弱いです。probe / heatmap による解釈性評価が必要です。

### EXP-006: baseline comparability FreshRetailNet

| model | best epoch | valid WAPE | test WAPE | MAE | RMSE | bias |
|---|---:|---:|---:|---:|---:|---:|
| `naive_recent_mean` | - | - | 0.5084 | 1.2112 | 2.2258 | 0.4704 |
| `naive_same_hour_recent_mean` | - | - | **0.3405** | **0.8112** | **1.3154** | 0.2714 |
| `feature_flatten_mlp` | 16 | 0.5664 | 0.5664 | 1.3493 | 3.7134 | -0.3507 |
| `proposed_with_decouple` | 3 | 0.6167 | 0.6167 | 1.4692 | 4.3314 | -0.3748 |

所見:

- FreshRetailNet では `naive_same_hour_recent_mean` が明確に最良です。
- proposed は feature MLP よりも悪く、naive baseline には大きく負けています。
- 予測性能で提案モデルを主張するには、現状の FreshRetailNet 設定では不十分です。
- まずは「hourly recent mean が強すぎる理由」を分析する必要があります。

### EXP-007: stockout / probe / heatmap synthetic

| model | best epoch | valid WAPE | test WAPE | MAE | RMSE | bias | z_global subgroup | z_day weekday | z_day holiday |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `proposed_stockout_mask` | 18 | 0.1006 | **0.0930** | **10.4837** | **14.9435** | 0.0100 | 0.5333 | 0.5804 | 0.7952 |
| `proposed_stockout_soft_0_1` | 24 | 0.0998 | 0.0931 | 10.4938 | 15.2102 | 0.0130 | 0.5083 | 0.5387 | 0.8286 |
| `proposed_stockout_uniform` | 20 | **0.0993** | 0.0934 | 10.5249 | 15.4440 | 0.0069 | 0.5167 | **0.7149** | **0.8830** |

所見:

- synthetic では3方式とも予測性能は非常に近いです。
- test WAPE / MAE / RMSE は `mask` が最良です。
- probe では `uniform` が weekday / holiday を最も強く捉えています。
- 欠品中の観測を完全に無視する方が予測性能はやや良く、欠品中も入れる方が calendar probe は強くなる傾向があります。

### EXP-007: stockout / probe / heatmap FreshRetailNet

| model | best epoch | valid WAPE | test WAPE | MAE | RMSE | bias | z_global subgroup | z_day weekday | z_day holiday |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `proposed_stockout_mask` | 2 | 0.6179 | 0.6179 | 1.4721 | 4.3482 | -0.4008 | 0.0000 | 0.1942 | **0.6577** |
| `proposed_stockout_soft_0_1` | 2 | 0.6188 | 0.6188 | 1.4742 | 4.3107 | -0.3246 | 0.0000 | **0.2031** | 0.5499 |
| `proposed_stockout_uniform` | 27 | **0.5588** | **0.5588** | **1.3312** | **3.4663** | -0.2955 | 0.2620 | 0.1973 | 0.4504 |

所見:

- FreshRetailNet では `uniform` が最良です。
- `uniform` は予測性能と bias の両方を改善しています。
- ただし naive baseline の `same_hour_recent_mean` にはまだ負けています。
- `probe_z_global_subgroup_accuracy` は majority accuracy 0.582 を下回っており、global latent が subgroup を十分に捉えているとは言えません。
- FreshRetailNet では probe の label overlap が train 9 classes / test 3 classes / overlap 3 classes で、評価が不安定です。

## 総合所見

1. `valid_wape` checkpoint と early stopping は機能しています。
2. synthetic では learned model が naive baseline を上回ります。
3. FreshRetailNet では naive_same_hour_recent_mean が非常に強く、現状の learned model は予測性能で負けています。
4. `proposed_with_decouple` は feature MLP と同等程度で、予測性能だけでは強い優位性を示せていません。
5. 欠品重みは synthetic では差が小さく、FreshRetailNet では `uniform` が最良でした。
6. probe は synthetic では一定の意味を持ちますが、FreshRetailNet では class imbalance / label overlap の問題が残ります。

## 次の判断

現時点で研究主張を組むなら、以下が妥当です。

```text
多粒度 local/global 分離は synthetic では naive baseline より有効であり、一部 probe でも日付要因を捉える。
一方、FreshRetailNet では強い same-hour recent mean baseline に対して予測性能で劣る。
したがって、現状の主張は予測性能優位ではなく、表現分解の可能性と、実データでの課題分析に置くべきである。
```

次に必要なのは以下です。

1. FreshRetailNet の `same_hour_recent_mean` が強い理由を、需要帯・欠品率・ゼロ率別に分析する。
2. 店舗×商品ではなく店舗×カテゴリ集約で再実験する。
3. proposed model に same-hour recent mean を residual baseline として組み込む。
4. 予測対象を「総売上」だけでなく「naive baseline からの残差」に変える。
5. probe 評価は subgroup ではなく、weekday / holiday / discount / hour pattern を中心に再設計する。

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
