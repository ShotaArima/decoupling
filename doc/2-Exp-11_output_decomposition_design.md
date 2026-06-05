# 2-Exp-11: Output-Decomposed Residual Representation

## 目的

これまでの 2-Exp-7 から 2-Exp-9 では、`z_global`, `z_day`, `z_hour`, `z_interaction` を分けて持たせても、最終的な `r_hat` は decoder がまとめて出力していた。

そのため、潜在表現が分かれていても、decoder 側で成分が混ざる余地が残っていた。2-Exp-11 ではこの点を直接検証するため、出力を次のように分ける。

```text
r_hat = r_global_hat + r_day_hat + r_hour_hat + r_interaction_hat
```

この設計により、各潜在表現がどの出力成分を担っているかを直接測る。

## 仮説

1. 出力も成分ごとに分けると、`z_day` と `z_hour` の役割が見えやすくなる。
2. day / hour / interaction に平均ゼロ制約を入れると、成分同士の混ざりが減る。
3. 真の成分を持つ synthetic では、各推定成分が対応する真の成分と相関する。
4. FreshRetailNet では真の成分は観測できないため、ablation と probe と下流補正で妥当性を見る。

## 追加したモデル

`OutputDecompositionResidualModel` を追加した。

このモデルは encoder を次の 4 系統に分ける。

- `z_global`: 系列全体の静的な残差傾向
- `z_day`: 曜日、休日、天候、販促などの日単位の変動
- `z_hour`: 時間帯ごとの形
- `z_interaction`: day と hour の組み合わせで初めて出る変動

decoder は各成分ごとに別 head を持つ。

```text
z_global      -> r_global_hat
z_day         -> r_day_hat
z_hour        -> r_hour_hat
z_interaction -> r_interaction_hat
```

最後に足し合わせて `r_hat` を作る。

## 追加した synthetic

`component_residual_retail` を追加した。

この dataset は、真の残差を次の形で生成する。

```text
r = r_global + r_day + r_hour + r_interaction + noise
```

さらに、学習・評価時に次の真値も batch に含める。

- `true_global`
- `true_day`
- `true_hour`
- `true_interaction`
- `true_residual`

これにより、推定した出力成分が本当に対応する真の成分を復元しているかを測れる。

## 評価指標

synthetic では次を主指標にする。

- `residual_mae`
- `residual_r2`
- `component_global_corr`
- `component_day_corr`
- `component_hour_corr`
- `component_interaction_corr`
- `component_*_mae`
- `component_ablation_without_*_mae_delta`
- `component_day_mean_abs`
- `component_hour_mean_abs`
- `component_interaction_day_mean_abs`
- `component_interaction_hour_mean_abs`

FreshRetailNet では真の成分がないため、次を見る。

- `baseline_cell_mae` と `corrected_cell_mae`
- high residual top 10% の改善
- latent probe
- component ablation
- 各成分の平均ゼロ制約が守られているか

## 実行コマンド

smoke:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-11_output_decomposition_smoke.json
```

synthetic 複数 seed:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-11_output_decomposition_synthetic.json
```

FreshRetailNet subset:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-11_output_decomposition_freshretailnet.json
```

## smoke の初期結果

`output_decomp_centered` は smoke で次の傾向を示した。

- `residual_mae`: 0.5867
- `baseline_cell_mae`: 0.6167
- `corrected_cell_mae`: 0.5867
- `component_global_corr`: 0.8140
- `component_day_corr`: 0.7355
- `component_hour_corr`: 0.9611
- `component_interaction_corr`: 0.2790
- `probe_z_day_weekday_accuracy`: 0.8095
- `probe_z_hour_hour_accuracy`: 0.4306

`output_decomp_no_center` と比べると、day / hour 成分の復元が大きく改善した。

これは、単に潜在表現を分けるだけでは不十分で、出力成分の制約まで含めた設計が必要であることを示す初期証拠になる。

## 論文化に向けた位置づけ

この実験は、論文の中心主張を支えるための controlled experiment として使う。

主張は次の形に整理できる。

1. 基準成分 `b` を取り除いた残差 `r` には、条件によって構造が残る。
2. 構造が残る場合、`r` は global / day / hour / interaction に分解して扱える。
3. ただし、分離表現を得るには latent を分けるだけでは弱い。
4. 出力成分と平均ゼロ制約を入れることで、分離の意味が強くなる。
5. FreshRetailNet では真の成分は観測できないため、synthetic で同定可能性を示し、実データで下流補正・probe・ablation を確認する。
