# 2-Exp-8: Structured Residual Synthetic

## 目的

これまでの synthetic では、`r = y - b` に weekday 以外の構造が弱く、多粒度分離の必要性を検証しにくい結果でした。
`2-Exp-8` では、同時間帯基準成分 `b` が強く、かつ残差 `r` に明示的な構造が残る synthetic dataset を追加します。

## 生成構造

`structured_residual_retail` は、売上を以下の形で生成します。

```text
y = baseline(series, hour) + residual_global(series)
  + residual_day(weekday, holiday, discount, weather)
  + residual_hour(subgroup, hour)
  + residual_interaction(discount x hour, holiday x hour)
  + noise
```

既存の `same_hour_recent_mean` で強い基準成分を取り除いた後も、weekday / discount / interaction の構造が残ることを狙います。

## 実行コマンド

Smoke:

```bash
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-8_structured_residual_diagnostics_smoke.json
uv run decoupled-ts residual-experiment --config configs/2-Exp-8_structured_residual_smoke.json
```

本実験:

```bash
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-8_structured_residual_diagnostics_synthetic.json
uv run decoupled-ts residual-experiment --config configs/2-Exp-8_structured_residual_synthetic.json
```

## Smoke 結果

診断では、`same_hour_recent_mean` が非常に強く、残差にも構造が残ることを確認しました。

| metric | value |
|---|---:|
| `same_hour_recent_mean.r2` | 0.9843 |
| `same_hour_recent_mean.residual_variance_ratio` | 0.0153 |
| `linear_probe_r2` | 0.6128 |
| `random_forest_probe_r2` | 0.7665 |
| `gradient_boosting_probe_r2` | 0.7114 |
| `group_variance_explained_weekday` | 0.4773 |
| `group_variance_explained_discount_bin` | 0.2045 |

この結果は、`b` が強く、かつ `r` に表現学習対象として十分な構造がある条件を作れていることを示します。

Smoke の残差表現学習では、2 epoch のため性能判断はしません。ただし runner は正常に完走し、`summary.csv`, `metrics.json`, latent `.npy`, `z_hour_heatmap.csv` が生成されます。

## 見るべき本実験指標

`summary.csv` で以下を比較します。

| 比較 | 目的 |
|---|---|
| `single_local_reference` vs `interaction_no_swap` | 自由度の高い local に対して、多粒度がどこまで近づくか |
| `day_hour_no_swap` vs `interaction_no_swap` | interaction の寄与 |
| `interaction_no_swap` vs `interaction_with_swap` | swap 正則化で分離性が改善するか |

主指標:

```text
residual_mae
residual_r2
corrected_cell_mae
probe_z_global_subgroup_accuracy
probe_z_day_weekday_accuracy
probe_z_day_discount_mae
probe_z_hour_hour_accuracy
swap_interaction_main_effect_delta
```
