# 次回ゼミ資料: 残差出力分解モデルの論拠整理

## 0. 今回のゼミで確認したいこと

今回の目的は、前回までの反省点を踏まえて、研究の主張を次の形に整理することである。

```text
時系列を global / local に分ける既存研究を出発点にする。
ただし、小売需要では local を単一の潜在表現として持つだけでは、
日・時間帯・日×時間帯の担当範囲が decoder 内で混ざる。

そこで本研究では、強い基準値で説明できる部分を先に取り除き、
残った residual correction を series / day / hour / interaction の
出力成分として分ける。
```

本研究で主張したいことは、「どの基準値でも常に予測精度が上がる」ことではない。
むしろ、基準値の後に残る残差構造を診断し、その構造が残る条件では、補正と解釈が同時に可能である、という条件付きの主張にする。

## 1. 前回からの反省点

前回までの議論では、元論文の global / local 表現分離から、本研究の 4 成分分解へ進む流れがやや飛んで見える可能性があった。

特に修正すべき点は次の 4 つである。

| 反省点 | 修正方針 |
|---|---|
| `global/local` から急に `series/day/hour/interaction` に分かれる | 小売需要では local の中に day, hour, day-hour interaction が混ざる、という問題として導入する |
| latent を分ければ解釈可能になるように見える | latent split だけでは decoder 内で混ざるため、出力分解が必要だと明示する |
| 強い baseline を常に超える主張に見える | 残差に構造が残る target で有効、という適用条件つきの主張にする |
| interaction 成分を実データで強く主張しすぎる | FreshRetailNet では hour 構造を主成功例、interaction は synthetic での検証を中心に置く |

したがって、次回ゼミでは「提案法が万能に精度を上げる」ではなく、次の論理に絞る。

1. 既存の global/local 分離は出発点として自然である。
2. 小売需要では local が day/hour/interaction に分かれる。
3. しかし latent を細かく分けるだけでは十分ではない。
4. 出力される residual correction そのものを成分分解し、centering constraints で担当範囲を固定する。
5. 成分が存在する条件では synthetic で回復でき、FreshRetailNet では hour 構造が残る target で補正と解釈が効く。

## 2. 研究の主張

### 2.1 問題設定

売上を $y_{i,d,h}$、基準値を $b_{i,d,h}$ とする。
本研究では、売上そのものではなく、基準値で説明できなかった残差を対象にする。

```math
r_{i,d,h} = y_{i,d,h} - b_{i,d,h}
```

この残差を、次の 4 成分に分けて予測する。

```math
\hat r_{i,d,h}
=
\hat g_i
+
\hat a_{i,d}
+
\hat c_{i,h}
+
\hat u_{i,d,h}
```

最終的な予測は次である。

```math
\hat y_{i,d,h}
=
b_{i,d,h}
+
\hat r_{i,d,h}
```

各成分の意味は次のように置く。

| 成分 | 意味 | 小売需要での読み方 |
|---|---|---|
| $\hat g_i$ | series 成分 | 店舗・商品系列に固有の残差 |
| $\hat a_{i,d}$ | day 成分 | その日全体の残差 |
| $\hat c_{i,h}$ | hour 成分 | 時間帯に対応する残差 |
| $\hat u_{i,d,h}$ | interaction 成分 | 特定の日の特定時間帯だけの残差 |

### 2.2 平均ゼロ制約の役割

出力を 4 成分に分けても、そのままでは成分の担当範囲が入れ替わる可能性がある。
例えば、本来 hour 成分として読みたい構造が interaction 成分に吸収されることがある。

そこで、day / hour / interaction に平均ゼロ制約を入れ、主効果と相互作用の担当範囲を固定する。

この制約の目的は、潜在表現が真の要因と一対一対応することを保証することではない。
目的は、出力成分が series / day / hour / interaction として検査可能な形にあることを保証することである。

## 3. 追加実験で分かったこと

### 3.1 Synthetic: 成分が存在する条件では回復できる

2-Exp-22 と 2-Exp-23 により、真の成分が分かる synthetic data では、`output_decomp_centered` が成分を高相関で回復できることが確認できた。

主な読み取りは次の通りである。

| 観点 | 分かったこと | 論文での使い方 |
|---|---|---|
| 成分回復 | `base` 条件で global/day/hour はほぼ完全に回復し、interaction も高く回復した | 提案法が成分構造を持つデータで正しく働く証拠 |
| centering の必要性 | `output_decomp_no_center` は residual MAE が大きく悪くなくても、global corr や interaction corr が崩れる | 予測誤差だけでは成分解釈を保証できない証拠 |
| interaction の必要性 | `high_interaction` では interaction なしモデルが悪化した | interaction 成分を入れる必要がある条件の証拠 |
| 失敗条件 | `small_sample` や `high_noise` では interaction corr が落ちる | 提案法の限界条件として使う |

この結果から言えることは、提案法が「成分が存在する条件で、その成分を出力空間で回復できる」ということである。

### 3.2 FreshRetailNet: `series_mean` residual では補正と hour 解釈が効く

FreshRetailNet では、基準値の選び方によって残差の性質が変わる。

`series_mean` を基準値にした場合、時間帯構造が残差に残り、提案法が効きやすい。
2-Exp-24 の `series_mean_12k` では、次の結果が得られた。

| model | baseline MAE | corrected MAE | top10 baseline | top10 corrected | hour corr |
|---|---:|---:|---:|---:|---:|
| `mae_grid_reference` | 0.0694 | 0.0494 | 0.2770 | 0.1994 | 0.9822 |
| `bias_constrained_001` | 0.0694 | 0.0487 | 0.2770 | 0.1964 | 0.9920 |

この結果から、`series_mean` residual では次が言える。

- 全体 MAE が改善する。
- high residual top10 でも改善する。
- hour component が residual hour profile と強く対応する。
- hour component を消すと MAE が悪化し、hour 成分が実際に補正に使われている。

### 3.3 FreshRetailNet: 系列数とブロックを変えても傾向は保たれる

2-Exp-24 では系列数を 2k / 6k / 12k に増やした。
2-Exp-25 では系列の開始位置を変え、block0 / block1 / block2 を比較した。

`series_mean` residual では、複数 block で改善が再現した。

| scenario | `mae_grid_reference` 改善 | `bias_constrained_001` 改善 | hour corr の範囲 |
|---|---:|---:|---:|
| `series_mean_block0_6k` | 0.0190 | 0.0186 | 0.9898-0.9957 |
| `series_mean_block1_6k` | 0.0173 | 0.0181 | 0.9766-0.9812 |
| `series_mean_block2_6k` | 0.0175 | 0.0179 | 0.8865-0.9754 |

これにより、`series_mean` residual の改善は先頭系列だけに依存しない、と説明できる。
ただし、FreshRetailNet 全体の完全な交差検証ではないため、論文では robustness check として扱う。

### 3.4 `same_hour_recent_mean_d7` は限界例になる

`same_hour_recent_mean_d7` は、直近 7 日の同時刻平均を基準値にする。
この基準値は時間帯構造を先に吸収するため、残差に hour 構造が残りにくい。

2-Exp-24 の `same_hour_recent_mean_d7_12k` では、次の結果だった。

| model | baseline MAE | corrected MAE | top10 baseline | top10 corrected | hour corr |
|---|---:|---:|---:|---:|---:|
| `mae_grid_reference` | 0.0574 | 0.0546 | 0.2530 | 0.2322 | -0.8861 |
| `bias_constrained_001` | 0.0574 | 0.0547 | 0.2530 | 0.2307 | -0.8738 |

MAE は少し改善するが、hour corr は負である。
したがって、この条件では「hour component が残差の時間帯構造を説明している」とは言いにくい。

これは失敗ではなく、提案法の適用条件を示す結果として扱う。

```text
強い same-hour baseline が時間帯構造を先に吸収すると、
残差出力分解の hour 成分は解釈しにくくなる。
```

### 3.5 Latent split だけでは足りない

2-Exp-27 では、残差 target にしただけで four-factor latent split が `global/local` より良くなるかを確認した。

結果として、`series_mean` residual を学習すること自体は有効だったが、latent を `global/day/hour/interaction` に分けるだけでは `global/local` reference を安定して上回らなかった。

| model | baseline cell MAE | corrected cell MAE | top10 corrected MAE | hour corr |
|---|---:|---:|---:|---:|
| `paper_global_local_residual` | 0.0721 | 0.0593 | 0.2794 | 0.8937 |
| `four_factor_global_day_hour_residual` | 0.0721 | 0.0671 | 0.2907 | 0.8908 |
| `four_factor_global_day_hour_interaction_residual` | 0.0721 | 0.0614 | 0.2989 | 0.8797 |

この結果から、次の主張は避ける。

```text
残差 target にすれば、four-factor latent split が global/local より明確に良くなる。
```

代わりに、次を主張する。

```text
残差 target には学習可能な構造が残る。
しかし、潜在表現を細分化するだけでは十分ではない。
出力成分そのものを分け、centering constraints で担当範囲を固定する必要がある。
```

### 3.6 Output decomposition + centering は latent split より説明しやすい

2-Exp-28 では、元論文に近い latent split と、本研究の output decomposition を同じ `series_mean` residual 上で比較した。

予測補正では、centered output decomposition 系が最も良かった。

| model | corrected MAE | corrected WAPE | residual R2 |
|---|---:|---:|---:|
| `paper_global_local_residual` | 0.060919 | 0.984469 | 0.0351 |
| `four_factor_global_day_hour_residual` | 0.061955 | 1.001208 | -0.0091 |
| `four_factor_global_day_hour_interaction_residual` | 0.061394 | 0.992142 | -0.0241 |
| `output_decomp_no_center` | 0.061685 | 0.996847 | 0.0173 |
| `output_decomp_centered_no_interaction` | 0.057189 | 0.924185 | 0.1984 |
| `output_decomp_centered` | 0.057247 | 0.925127 | 0.1855 |

高残差 top10 でも、centered output decomposition は改善した。

| model | high residual baseline | high residual corrected |
|---|---:|---:|
| `paper_global_local_residual` | 0.292300 | 0.292751 |
| `output_decomp_no_center` | 0.292300 | 0.295369 |
| `output_decomp_centered_no_interaction` | 0.292300 | 0.268087 |
| `output_decomp_centered` | 0.292300 | 0.265624 |

hour profile でも、centered output decomposition が安定した。

| model | residual hour profile corr | hour component residual profile corr |
|---|---:|---:|
| `paper_global_local_residual` | 0.8893 | n/a |
| `output_decomp_no_center` | 0.8574 | 0.8469 |
| `output_decomp_centered_no_interaction` | 0.9962 | 0.9944 |
| `output_decomp_centered` | 0.9923 | 0.9919 |

この実験は、次の論理を支える。

```text
latent を分けるだけでは、成分の意味や補正性能は安定しない。
output decomposition + centering により、補正性能と成分解釈を同時に評価できる。
```

## 4. 添付資料から整理できる論拠

添付資料と既存メモから、関連研究の論拠は次のように整理できる。

| 論拠 | 支える資料・研究群 | 本研究での使い方 |
|---|---|---|
| 時系列には時間不変な global 情報と、時変な local 情報がある | Decoupling Local and Global Representations, FHVAE, DSVAE, C-DSVAE | 元論文の問題意識を出発点にする |
| local/global の潜在分離は有効だが、出力成分の意味までは保証しない | Decoupling Local and Global Representations, C-DSVAE, DSVAE | latent split だけでは不十分という導入に使う |
| forecasting では global 構造と local 補正を分ける考えが有効 | Deep Factors, DeepGLO | 強い baseline と residual correction の考えを正当化する |
| trend/season や multi-scale decomposition は予測に効く | CoST, Autoformer, FEDformer, TimeMixer | 「分けること」が予測にも意味を持つことを支える |
| 既存研究の多くは coarse な二分割か latent decomposition に留まる | local/global, static/dynamic, trend/season 系の研究 | 本研究の差分を output-level operational decomposition として示す |
| 解釈可能性は latent の名前ではなく、検査可能な出力・ablation・profile 対応で示すべき | synthetic recovery, component ablation, hour profile corr | 実験設計の妥当性を支える |

この整理から、本研究の related work で言うべき差分は次である。

```text
既存研究は、時系列表現を local/global, static/dynamic, trend/season などに分ける。
しかし、多くは潜在空間上の分離であり、出力された補正量が
どの運用単位に対応するかは必ずしも明示されない。

本研究は、強い基準値の後に残る residual correction を対象にし、
その出力を series/day/hour/interaction という小売運用上の単位へ分ける。
さらに centering constraints と ablation/profile evaluation によって、
各成分がどの担当範囲を持つかを検査可能にする。
```

## 5. 次回発表の構成案

### Slide 1: 研究の問い

```text
強い基準値でかなり説明できる小売需要に対して、
残った residual をどのように分解すれば、
予測補正と解釈を同時に扱えるか。
```

### Slide 2: 元論文からの出発点

- 元論文は global / local 表現を分ける。
- 小売では global は系列固有の水準、local は日・時間帯の変動に対応する。
- ただし local を単一表現にすると、day / hour / interaction が混ざる。

### Slide 3: 前回からの修正点

- 主張を「常に強い baseline を超える」から「残差構造が残る条件で有効」に修正する。
- latent split ではなく output decomposition を主張の中心にする。
- FreshRetailNet では hour 構造を主成功例にする。
- interaction は synthetic で必要性を示し、実データでは強く主張しすぎない。

### Slide 4: 提案手法

```math
\hat r_{i,d,h}
=
\hat g_i
+
\hat a_{i,d}
+
\hat c_{i,h}
+
\hat u_{i,d,h}
```

- 出力を series / day / hour / interaction に分ける。
- centering constraints で成分の担当範囲を固定する。
- 最終予測は $b+\hat r$ とする。

### Slide 5: Synthetic の証拠

- 成分が存在する条件で `output_decomp_centered` が回復できる。
- centering なしでは residual MAE がよくても成分解釈が崩れる。
- interaction が強い条件では interaction 成分が必要になる。
- noise / small sample では限界が出る。

### Slide 6: FreshRetailNet の主成功例

- `series_mean` residual では補正と hour 解釈が効く。
- `series_mean_12k` で baseline MAE 0.0694 から corrected MAE 0.0487-0.0494。
- high residual top10 も 0.2770 から 0.1964-0.1994。
- hour corr は 0.9822-0.9920。

### Slide 7: Robustness

- 系列数 2k / 6k / 12k で傾向が保たれる。
- block0 / block1 / block2 でも `series_mean` residual の改善が再現する。
- ただし FreshRetailNet 全体の完全な交差検証ではないため、robustness check として扱う。

### Slide 8: 限界例

- `same_hour_recent_mean_d7` は時間帯構造を先に吸収する。
- MAE は少し改善するが、hour corr は負になる。
- これは「提案法が失敗した」というより、「残差に hour 構造が残らないと解釈性が落ちる」という適用条件を示す。

### Slide 9: Latent split との比較

- 2-Exp-27 では residual target でも four-factor latent split は `global/local` を明確に上回らなかった。
- 2-Exp-28 では output decomposition + centering が latent split よりよい補正と hour profile 対応を示した。
- したがって、本研究の主張は latent split ではなく output decomposition に置く。

### Slide 10: 現時点の論文ライン

```text
Synthetic:
成分が存在する条件で output decomposition + centering は成分を回復できる。

FreshRetailNet:
series_mean residual のように hour 構造が残る条件では、
予測補正・high residual 改善・hour component 解釈が成立する。

Limitation:
same-hour baseline のように時間帯構造を先に吸収する基準値では、
残差成分の解釈可能性は低下する。
```

## 6. ゼミで相談したい点

1. 実データの主張を `hour structure` 中心に絞る方針でよいか。
2. `interaction` は synthetic で必要性を示し、FreshRetailNet では強く主張しない方針でよいか。
3. `same_hour_recent_mean_d7` を失敗例ではなく、適用条件を示す限界例として置く説明でよいか。
4. latent split との比較は 2-Exp-27 / 2-Exp-28 を使い、「latent を分けるだけでは足りない」と主張してよいか。
5. 本文の主表は synthetic recovery、FreshRetailNet correction、statistical validation に絞り、block robustness と latent comparison は補助表に回すべきか。

## 7. 発表後にやること

| 優先度 | 作業 | 目的 |
|---|---|---|
| 高 | Method 節を output decomposition 中心に書き直す | 主張の中心を固定する |
| 高 | Related Work を local/global latent から residual output decomposition へつなぐ | 既存研究との差分を明確にする |
| 高 | FreshRetailNet の主表を `series_mean` と `same_hour_recent_mean_d7` の対比に絞る | 成功条件と限界条件を同時に示す |
| 中 | 2-Exp-28 を補助表として整理する | latent split だけでは足りない証拠にする |
| 中 | block robustness を appendix に置く | 系列選択依存の懸念を弱める |
| 低 | interaction の実データ主張を追加で強める実験 | 現時点では必須ではない |
