# EXP-008〜EXP-012: FreshRetailNet 向け次段階実験計画

## 背景

`EXP-004〜EXP-007` の結果では、synthetic data では学習モデルが naive baseline を上回った一方、FreshRetailNet では `naive_same_hour_recent_mean` が最も強い結果になりました。

したがって、次の段階で単純にモデルを複雑化するのは優先度が低いです。まず確認すべきことは、以下です。

```text
なぜ FreshRetailNet では same-hour recent mean が強いのか。
提案モデルはその強い短期・同時刻パターンを無視していないか。
多粒度 latent は、総需要そのものではなく baseline からの残差を説明する方が適切ではないか。
```

このドキュメントでは、次に実施する実験を `EXP-008` から `EXP-012` として整理します。

## 全体方針

次の5つを順番に確認します。

| Experiment | 目的 |
|---|---|
| EXP-008 | FreshRetailNet の `same_hour_recent_mean` が強い理由を分析する |
| EXP-009 | 店舗×カテゴリ単位に集約して再実験する |
| EXP-010 | proposed model に same-hour recent mean を residual baseline として組み込む |
| EXP-011 | 予測対象を総売上ではなく naive baseline からの残差に変える |
| EXP-012 | probe を subgroup から weekday / holiday / discount / hour pattern 中心に再設計する |

## EXP-008: same-hour recent mean が強い理由の分析

### 問い

```text
FreshRetailNet で naive_same_hour_recent_mean が強いのは、
短期自己相関、同時刻周期性、低需要・ゼロ過多、欠品、または評価分割のどれが原因か？
```

### 実験内容

モデルを増やすのではなく、既存の `predictions.csv` と元データから誤差を分解します。

確認する観点:

| 分析軸 | 見るもの |
|---|---|
| 需要帯別 | target を quantile 分割し、各帯で WAPE / bias を計算 |
| ゼロ率別 | history sales zero rate ごとに WAPE / bias を計算 |
| 欠品率別 | history stockout rate ごとに WAPE / bias を計算 |
| 同時刻安定性 | same-hour recent mean と target の相関 |
| 直近日依存 | last-day / recent-mean / same-hour mean の差 |
| 系列単位 | store-product ごとに naive が強い系列と弱い系列を分ける |

### 必要な入力

既存実験の出力:

```text
runs/EXP-006_baseline_comparability_freshretailnet/*/predictions.csv
runs/EXP-006_baseline_comparability_freshretailnet/data_audit.json
runs/EXP-006_baseline_comparability_freshretailnet/summary.csv
```

追加で必要になる可能性があるもの:

```text
series_id
history zero rate
history stockout rate
target quantile
```

現状の `predictions.csv` には `series_id` や需要帯情報が入っていないため、EXP-008 では prediction export の拡張が必要です。

### 成果物

```text
runs/EXP-008_same_hour_analysis/summary.csv
runs/EXP-008_same_hour_analysis/summary.json
runs/EXP-008_same_hour_analysis/all_predictions_with_features.csv
runs/EXP-008_same_hour_analysis/error_by_target_quantile.csv
runs/EXP-008_same_hour_analysis/error_by_zero_rate.csv
runs/EXP-008_same_hour_analysis/error_by_stockout_rate.csv
runs/EXP-008_same_hour_analysis/error_by_store_id.csv
runs/EXP-008_same_hour_analysis/error_by_second_category_id.csv
runs/EXP-008_same_hour_analysis/error_by_third_category_id.csv
runs/EXP-008_same_hour_analysis/model_pairwise_comparison.csv
runs/EXP-008_same_hour_analysis/correlations.json
doc/EXP-008_same_hour_analysis_summary.md
```

### 実装済みコマンド

Synthetic:

```bash
uv run decoupled-ts same-hour-analysis --config configs/EXP-008_same_hour_analysis_synthetic.json
```

FreshRetailNet:

```bash
uv run decoupled-ts same-hour-analysis --config configs/EXP-008_same_hour_analysis_freshretailnet.json
```

### 判断基準

`same_hour_recent_mean` が特定の条件で強いなら、次のモデル改善方針が決まります。

| 結果 | 次の対応 |
|---|---|
| 低需要・ゼロ過多で強い | 店舗×カテゴリ集約、または zero-inflated modeling |
| 欠品率が高いほど強い | 欠品処理を見直す |
| 同時刻相関が高い | residual baseline 化を優先 |
| 高需要帯で弱い | high-demand 専用の誤差改善を狙う |

## EXP-009: 店舗×カテゴリ単位への集約

### 問い

```text
店舗×商品では疎すぎるため、day/hour latent が安定して学習できていないのではないか？
```

### 実験内容

サンプル単位を `store_id × product_id` から、より粗い単位に変えます。

候補:

| サンプル単位 | 優先度 | 理由 |
|---|---:|---|
| `store_id × third_category_id` | A | 商品粒度より安定し、カテゴリ解釈も残る |
| `store_id × second_category_id` | B | さらに安定するが粗い |
| `city_id × third_category_id` | C | 店舗固有性は弱まるが疎性は改善 |

### 比較対象

最低限、以下を比較します。

```text
naive_same_hour_recent_mean
feature_flatten_mlp
global_only
proposed_with_decouple
proposed_stockout_uniform
```

### 期待される結果

```text
集約によりゼロ率・欠品影響が下がり、proposed model の WAPE / bias が改善する。
特に z_day / z_hour probe が安定する。
```

### 失敗した場合の解釈

```text
疎性だけが原因ではなく、モデルが same-hour の短期周期性を十分に使えていない。
この場合は EXP-010 / EXP-011 の residual baseline 化を優先する。
```

### 成果物

```text
runs/EXP-009_store_category_aggregation/summary.csv
runs/EXP-009_store_category_aggregation/data_audit.json
runs/EXP-009_store_category_aggregation/*/predictions.csv
runs/EXP-009_store_category_aggregation/*/metrics.json
doc/EXP-009_store_category_aggregation_summary.md
```

## EXP-010: same-hour recent mean を residual baseline として組み込む

### 問い

```text
提案モデルは、総需要をゼロから予測するより、
強い naive baseline を補正する residual model として使う方が適切ではないか？
```

### モデル案

現在:

```text
y_hat = f(z_global, z_day, z_hour)
```

変更案:

```text
baseline = same_hour_recent_mean(X)
residual = f(z_global, z_day, z_hour)
y_hat = baseline + residual
```

必要に応じて、非負制約を入れます。

```text
y_hat = relu(baseline + residual)
```

または residual の大きさを安定化します。

```text
y_hat = baseline * exp(delta)
```

### 比較対象

```text
naive_same_hour_recent_mean
feature_flatten_mlp
proposed_with_decouple
proposed_residual_additive
proposed_residual_multiplicative
```

### 期待される結果

```text
same-hour naive の強さを維持しつつ、
販促・休日・天気・欠品などで naive が外すケースを latent model が補正する。
```

### 成果物

```text
runs/EXP-010_residual_baseline/summary.csv
runs/EXP-010_residual_baseline/*/predictions.csv
runs/EXP-010_residual_baseline/*/metrics.json
runs/EXP-010_residual_baseline/*/history.jsonl
doc/EXP-010_residual_baseline_summary.md
```

## EXP-011: 予測対象を naive baseline からの残差に変える

### 問い

```text
総売上を直接予測するより、
same-hour recent mean からの残差を予測した方が、
day/hour latent の役割が明確になるのではないか？
```

### 実験内容

予測対象を以下に変更します。

```text
target_residual = y_true - same_hour_recent_mean
```

モデルは residual を予測します。

```text
residual_hat = f(z_global, z_day, z_hour)
y_hat = same_hour_recent_mean + residual_hat
```

評価は最終的な `y_hat` に対して行います。

### EXP-010 との違い

| 実験 | 違い |
|---|---|
| EXP-010 | モデル出力を baseline に足すが、loss は最終予測で最適化 |
| EXP-011 | 教師信号自体を residual に変え、latent が補正成分を学ぶことを明示 |

### 期待される結果

```text
予測性能が naive baseline を上回る。
また、z_day probe が holiday / discount をより強く捉える。
```

### 成果物

```text
runs/EXP-011_residual_target/summary.csv
runs/EXP-011_residual_target/*/predictions.csv
runs/EXP-011_residual_target/*/metrics.json
runs/EXP-011_residual_target/*/latent_diagnostics.json
doc/EXP-011_residual_target_summary.md
```

## EXP-012: probe の再設計

### 問い

```text
subgroup probe は FreshRetailNet で class imbalance / label overlap の問題が大きい。
より研究仮説に近い probe に置き換えるべきではないか？
```

### 現状の問題

FreshRetailNet では以下が起きています。

```text
probe_subgroup_train_classes = 9
probe_subgroup_test_classes = 3
probe_subgroup_overlap_classes = 3
probe_subgroup_majority_accuracy = 0.582
```

この状態では、`z_global` から subgroup を当てる probe は安定した評価になりにくいです。

### 新しい probe

| latent | probe target | 評価指標 | 期待 |
---|---|---:|---|
| `z_day` | weekday | accuracy | 高い |
| `z_day` | holiday_flag | accuracy / AUROC | 高い |
| `z_day` | discount | MAE / R2 | 低 MAE |
| `z_day` | weather | MAE / R2 | 低 MAE |
| `z_hour` | hour index | accuracy | 高い |
| `z_hour` | morning/lunch/evening/night cluster | accuracy | 高い |
| `z_global` | average demand bin | accuracy | 高い |
| `z_global` | category id | accuracy | subgroup より安定すれば採用 |

### 可視化

以下を出力します。

```text
z_hour_heatmap.csv
z_hour_heatmap.png
z_day_by_weekday.csv
z_day_by_holiday.csv
z_global_by_demand_bin.csv
```

### 成果物

```text
runs/EXP-012_probe_redesign/summary.csv
runs/EXP-012_probe_redesign/*/probe_metrics.json
runs/EXP-012_probe_redesign/*/z_hour_heatmap.csv
runs/EXP-012_probe_redesign/*/z_hour_heatmap.png
doc/EXP-012_probe_redesign_summary.md
```

## 推奨実行順序

優先順位は以下です。

1. EXP-008: same-hour recent mean の強さを分析する。
2. EXP-010: same-hour baseline を proposed model に組み込む。
3. EXP-011: residual target 化を試す。
4. EXP-009: 店舗×カテゴリ集約で再実験する。
5. EXP-012: probe と heatmap を研究発表向けに整える。

理由:

```text
現状の最大課題は、FreshRetailNet で naive_same_hour_recent_mean に負けていること。
したがって、まず naive の強さを分析し、それをモデルに組み込む方が、
モデルをさらに複雑化するより優先度が高い。
```

## 研究上の位置づけ

次段階の研究主張は、以下のどちらかに収束させます。

### 予測性能を改善できた場合

```text
小売需要時系列では same-hour recent mean が強い baseline である。
本研究では、この強い短期周期 baseline を residual として組み込み、
その補正成分を multi-grain latent representation で説明することで、
予測性能と解釈性を両立する。
```

### 予測性能で naive に勝てない場合

```text
FreshRetailNet では短期同時刻平均が非常に強く、
総需要予測では深層 latent model の優位性は限定的だった。
一方で、multi-grain latent は weekday / holiday / discount / hour pattern などの要因を分離して捉える可能性があり、
予測性能単独ではなく、要因分解・解釈性の観点から評価する必要がある。
```
