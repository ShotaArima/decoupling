# 2-Exp-11: Four-Factor Residual Representation

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

## 実行結果

実行順:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-11_output_decomposition_smoke.json
uv run decoupled-ts residual-sweep --config configs/2-Exp-11_output_decomposition_synthetic.json
uv run decoupled-ts residual-sweep --config configs/2-Exp-11_output_decomposition_freshretailnet.json
```

### smoke

smoke では `output_decomp_centered` が最も良かった。

| model | residual MAE | corrected cell MAE | day corr | hour corr | interaction corr |
|---|---:|---:|---:|---:|---:|
| latent concat interaction | 0.6023 | 0.6023 | - | - | - |
| output decomp no center | 0.6059 | 0.6059 | 0.1654 | 0.5341 | -0.0014 |
| output decomp centered | 0.5863 | 0.5863 | 0.7426 | 0.9609 | 0.2718 |

平均ゼロ制約を入れた `output_decomp_centered` では、`day/hour` 成分の復元が明確に改善した。特に `hour` は高い相関を示している。

### synthetic 複数 seed

5 seed の synthetic では、`output_decomp_centered` が最も強い結果だった。

| model | runs | residual MAE mean | residual R2 mean | corrected cell MAE mean | top10 corrected MAE mean |
|---|---:|---:|---:|---:|---:|
| latent concat interaction | 5 | 0.0350 | 0.9928 | 0.0350 | 0.0512 |
| output decomp no center | 5 | 0.0403 | 0.9913 | 0.0403 | 0.0556 |
| output decomp centered | 5 | 0.0177 | 0.9989 | 0.0177 | 0.0250 |

`output_decomp_centered` の成分復元:

| component | corr mean | MAE mean |
|---|---:|---:|
| global | 0.9995 | 0.0090 |
| day | 0.9993 | 0.0080 |
| hour | 0.9997 | 0.0106 |
| interaction | 0.9879 | 0.0065 |

ablation でも各成分を消したときに MAE が増えている。

| removed component | MAE delta mean |
|---|---:|
| global | +0.2703 |
| day | +0.2051 |
| hour | +0.5171 |
| interaction | +0.0293 |

この結果は、真の成分がある controlled setting では、4変数への分解と平均ゼロ制約によって各成分をほぼ正しく回収できることを示している。

### FreshRetailNet subset

FreshRetailNet subset では、synthetic と同じ改善は出なかった。

| model | runs | baseline cell MAE mean | corrected cell MAE mean | residual R2 mean | top10 corrected MAE mean |
|---|---:|---:|---:|---:|---:|
| latent concat interaction | 3 | 0.0547 | 0.0566 | -0.0057 | 0.2444 |
| output decomp centered | 3 | 0.0547 | 0.0591 | -0.0595 | 0.2487 |

`output_decomp_centered` は平均ゼロ制約自体は守れている。

| diagnostic | mean |
|---|---:|
| component_day_mean_abs | 2.94e-9 |
| component_hour_mean_abs | 3.61e-9 |
| component_interaction_day_mean_abs | 1.67e-9 |
| component_interaction_hour_mean_abs | 1.20e-11 |

しかし、補正は悪化した。特に `corrected_cell_bias_mean = -0.3072` と大きく、実データでは global 成分または補正値の平均が崩れている可能性がある。

ablation でも day/hour/interaction を消したときの悪化はほぼゼロであり、実データ subset ではこれらの成分が有効な補正情報として使われていない。

| removed component | MAE delta mean |
|---|---:|
| global | -0.0034 |
| day | +0.0001 |
| hour | +0.0004 |
| interaction | +0.0000 |

## 現時点の解釈

2-Exp-11 から得られた結論は次の通り。

1. synthetic では仮説はかなり強く支持された。
2. latent を分けるだけでなく、出力も成分ごとに分け、day/hour/interaction に平均ゼロ制約を入れることが重要だった。
3. FreshRetailNet subset では同じ設計をそのまま適用しても改善しなかった。
4. FreshRetailNet 側では、残差に十分な構造がない、subset の切り方が合っていない、または補正値の bias 制御が弱い可能性がある。
5. したがって、次はモデルを大きくするより先に「この分解が効く条件」を特定する必要がある。

## 次に行うべき実験

### 2-Exp-12: FreshRetailNet の適用条件を切り分ける

目的は、FreshRetailNet のどの subset なら残差補正が意味を持つかを調べること。

候補 subset:

- residual variance が大きい系列
- gradient boosting probe で residual 構造が読める系列
- 非ゼロ売上が一定以上ある系列
- discount / stockout / holiday などの外生変数が十分に変動する系列
- high residual が特定曜日・時間帯に偏る系列

比較する指標:

- baseline cell MAE
- corrected cell MAE
- high residual top 10% MAE
- residual R2
- residual probe R2
- component ablation delta

成功条件:

- 全体平均ではなくても、構造が強い subset で `b + r_hat` が `b` を上回る。
- 構造が弱い subset では改善しない、という負例も同時に示せる。

### 2-Exp-13: bias 制御を入れた補正実験

FreshRetailNet では `output_decomp_centered` の `corrected_cell_bias` が大きく崩れた。

次は以下を比較する。

- post-hoc に validation residual mean を引く
- loss に residual bias penalty を入れる
- series 単位で補正平均を 0 に近づける
- global head の出力範囲を制限する

成功条件:

- corrected bias が baseline より悪化しない。
- corrected cell MAE が少なくとも baseline と同水準になる。
- high residual top 10% で改善が出る。

### 2-Exp-14: synthetic の難易度 sweep

論文化には「どの条件で分離できるか」が必要になる。

次は synthetic で以下を sweep する。

- interaction scale
- noise scale
- series 数
- 日数
- 欠品率
- baseline が強すぎる場合と弱い場合

成功条件:

- 成分分離が可能な条件と難しい条件を表で示せる。
- sample size や noise に対して、成分相関がどのように落ちるかを示せる。

### 2-Exp-15: 論文用の最終比較

最終的には次の比較表が必要。

- baseline only
- latent concat interaction
- output decomp no center
- output decomp centered
- output decomp centered + bias control
- output decomp centered + selected FreshRetailNet subset

この比較を synthetic と FreshRetailNet で同じ形式にそろえる。

論文の主張は「常に FreshRetailNet 全体で改善する」ではなく、次の形に寄せるのがよい。

> 残差に構造が存在し、成分の平均ゼロ条件が妥当なとき、4変数への分解を持つ residual representation は成分を回収できる。実データではこの条件を満たす subset で補正効果を検証する必要がある。

## 論文化に向けた位置づけ

この実験は、論文の中心主張を支えるための controlled experiment として使う。

主張は次の形に整理できる。

1. 基準成分 `b` を取り除いた残差 `r` には、条件によって構造が残る。
2. 構造が残る場合、`r` は global / day / hour / interaction に分解して扱える。
3. ただし、分離表現を得るには latent を分けるだけでは弱い。
4. 出力成分と平均ゼロ制約を入れることで、分離の意味が強くなる。
5. FreshRetailNet では真の成分は観測できないため、synthetic で同定可能性を示し、実データで下流補正・probe・ablation を確認する。
