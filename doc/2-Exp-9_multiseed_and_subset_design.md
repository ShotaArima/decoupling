# 2-Exp-9: Multi-Seed and FreshRetailNet Subset Evaluation

## 目的

`2-Exp-8` では、真の residual 構造を持つ synthetic data 上で `interaction_no_swap` と `interaction_with_swap` が中心候補になった。
`2-Exp-9` では、この結論が seed 依存ではないかを確認し、さらに FreshRetailNet の中でも residual 構造が強い subset に限定して同じ比較を行う。

## 検証するモデル

- `interaction_no_swap`
  - `z_global`, `z_day`, `z_hour`, `z_interaction` を使う。
  - swap 正則化なし。
- `interaction_with_swap`
  - 同じ表現構造。
  - counterfactual-style latent swap 正則化あり。

この段階では単一 local や global/day/hour の広い ablation には戻らず、`2-Exp-8` で残った中心候補だけを複数 seed で比較する。

## 実験コマンド

Smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-9_multiseed_structured_residual_smoke.json
```

Structured residual synthetic の本実験:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-9_multiseed_structured_residual_synthetic.json
```

FreshRetailNet subset:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-9_freshretailnet_subset_interaction.json
```

## FreshRetailNet subset の切り出し

`configs/2-Exp-9_freshretailnet_subset_interaction.json` では `dataset.subset_filter` を使う。
各系列について same-hour recent mean baseline から residual を作り、以下の条件で系列を残す。

- `min_nonzero_rate`: ゼロ売上ばかりの系列を除く。
- `min_observed_rate`: 欠品・未観測が多すぎる系列を除く。
- `min_residual_std`: residual がほぼ無構造な系列を除く。
- `sort_by = residual_std`, `top_k`: residual 変動が大きい系列を優先する。

これにより「FreshRetailNet 全体では baseline が強すぎるが、一部の外れ・変動系列では residual 表現が効くか」を見る。

## 出力

`residual-sweep` は `sweep.output_dir` 配下に以下を出力する。

- `base_config.json`: sweep 元 config。
- `seed_<seed>_config.json`: seed ごとに展開された config。
- `seed_<seed>/`: 通常の `residual-experiment` 出力。
- `all_results.csv`: seed x model の全結果。
- `aggregate.csv`: model ごとの mean/std 集計。
- `summary.json`: JSON 版の全結果と集計。

## 主指標

再構成・補正:

- `residual_mae`, `residual_rmse`, `residual_r2`
- `corrected_cell_mae` vs `baseline_cell_mae`
- `high_residual_top10_corrected_mae` vs `high_residual_top10_baseline_mae`

分離性:

- `probe_z_global_subgroup_accuracy`
- `probe_z_day_weekday_accuracy`
- `probe_z_hour_hour_accuracy`
- `probe_z_day_discount_mae`
- `swap_interaction_main_effect_delta`

## 判断基準

- Structured residual synthetic で mean/std を見ても `interaction_no_swap` / `interaction_with_swap` の差が小さいなら、再構成性能は同水準と扱う。
- swap ありで probe または swap diagnostic が安定して改善するなら、分離性を目的に swap 正則化を採用する余地がある。
- FreshRetailNet subset で `corrected_cell_mae` または `high_residual_top10_corrected_mae` が baseline より改善するなら、全体平均ではなく「外れケース補正」として仮説を立て直せる。
- FreshRetailNet subset でも改善がない場合は、residual 表現の学習以前に `r` の定義、subset 条件、または補正対象の設計を見直す。
