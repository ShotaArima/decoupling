# 2-Exp-7: 反実仮想 swap 正則化による分離性検証

## 目的

`2-Exp-2〜6` では、`z_global + z_day + z_hour` の構造は入っているものの、分離を強制する制約は covariance penalty が中心でした。
この実験では、反実仮想的に latent を swap したときの変化方向を制約し、分離性が改善するか確認します。

## 実行コマンド

Smoke:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-7_swap_regularization_smoke.json
```

Synthetic:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-7_swap_regularization_synthetic.json
```

FreshRetailNet:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-7_swap_regularization_freshretailnet.json
```

## 比較

| variant | 目的 |
|---|---|
| `single_local_reference` | 自由度が高い single local の参照 |
| `day_hour_no_swap` | `z_global + z_day + z_hour`、swap 正則化なし |
| `day_hour_with_swap` | `z_global + z_day + z_hour`、swap 正則化あり |
| `interaction_no_swap` | `z_interaction` あり、swap 正則化なし |
| `interaction_with_swap` | `z_interaction` あり、swap 正則化あり |

## 追加した正則化

各 batch 内で latent を別サンプルのものに交換し、以下の不変性を弱く課します。

| swap | 制約 |
|---|---|
| `z_global` swap | cell shape の中心化後差分が大きくなりすぎない |
| `z_day` swap | day swap 後も hour profile の平均が変わりすぎない |
| `z_hour` swap | hour swap 後も day profile の平均が変わりすぎない |
| `z_interaction` swap | interaction が additive main effect を持ちすぎない |

これは完全な反実仮想生成ではなく、まずは分離性改善を見るための弱い swap consistency loss です。

## 見る指標

再構成・補正:

```text
residual_mae
residual_r2
corrected_cell_mae
high_residual_top10_corrected_mae
```

probe:

```text
probe_z_global_subgroup_accuracy
probe_z_day_weekday_accuracy
probe_z_day_discount_mae
probe_z_hour_hour_accuracy
```

swap diagnostics:

```text
swap_global_total_delta
swap_global_shape_delta
swap_day_total_delta
swap_day_hour_profile_delta
swap_hour_total_delta
swap_hour_day_profile_delta
swap_interaction_total_delta
swap_interaction_main_effect_delta
```

期待は、swap 正則化によって再構成性能を大きく落とさず、probe と swap diagnostics の役割分担が改善することです。
