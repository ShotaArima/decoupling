# Proposal: 基準値からのズレを説明する残差成分モデル

## 目的

この proposal は、これまでの `2-Exp-*` 系列で見えた課題を踏まえ、修士論文または論文 1 本として成立する形に研究を収束させるための計画である。

これまでの実験では、売上そのものではなく、強い基準値から外れた部分、つまり残差を学習対象にした。

売上を $y$、基準値を $b$、残差を $r$ とすると、

$$
r_{i,d,h}=y_{i,d,h}-b_{i,d,h}
$$

である。

ここで、

- $i$: 店舗・商品系列
- $d$: 日
- $h$: 時間帯
- $b$: same-hour recent mean などの基準値
- $r$: 基準値では説明できなかったズレ

である。

これまでの結果から、FreshRetailNet では $b$ の選び方で残差の性質が大きく変わることが分かった。

`same_hour_recent_mean` 系の $b$ は非常に強く、残差から hour 構造をかなり取り除く。そのため、単純に $r$ を学習しても、平均的な予測補正では大きな改善は出にくい。

一方、`series_mean` 系の $b$ では hour 構造が残り、`b+\hat r` による補正と high residual top10 の改善が安定して確認できた。

合成データでは、残差に明確な構造がある場合、`global/day/hour/interaction` の成分分解は成立した。特に `output_decomp_centered` は、真の成分がある synthetic で global/day/hour をほぼ完全に回復し、interaction も高い相関で回復した。

したがって次の研究の中心は、単に latent を分けることではなく、**基準値からのズレを、出力成分として読める形に分けること**に置く。

なお、2-Exp-26 では、元論文に近い `global + local` 分解を通常の売上予測に移植し、local を `day/hour/interaction` に分ける比較も行った。FreshRetailNet では `global + day + hour` が `global + local` より WAPE を 0.6233 から 0.6133 に小幅改善した一方、interaction まで加えると WAPE は 0.6267 となり、通常予測では安定した改善にならなかった。

この結果は、day/hour 分割が小売データに合う導入であることを示す一方で、売上全体を直接 4 成分に分けるだけでは interaction の意義が見えにくいことも示している。そのため本研究では、基準値で説明できる主効果を除いた残差に対して、day/hour/interaction の表現学習と出力分解を行う。

さらに 2-Exp-27 では、`series_mean` residual に対して同じ latent split を比較した。全モデルが baseline MAE 0.0721 を補正したが、最も良かったのは `global/local` residual reference で corrected MAE 0.0593 だった。day/hour split は 0.0671、interaction 付きは 0.0614 であり、潜在表現を細分化するだけでは十分ではなかった。

2-Exp-28 では、同じ `series_mean` residual 上で latent split 系と output decomposition 系を直接比較した。
centered output decomposition は corrected MAE 0.0572 まで改善し、latent split 系の最良である `paper_global_local_residual` の 0.0609 を上回った。
高残差上位 10% でも、latent split 系は baseline 0.2923 に対して同等または悪化したが、`output_decomp_centered` は 0.2656、`output_decomp_centered_no_interaction` は 0.2681 まで改善した。
このため、主提案を latent split ではなく output decomposition + centering に置く論理は、Exp-28 によって強められる。

このため、proposal の中心は「latent を細かく分けること」ではなく、**残差出力そのものを global/day/hour/interaction 成分に分け、制約によって各成分の意味を固定すること**に置く。

## 中心仮説

残差 $r_{i,d,h}$ は、次のような成分に分解できる。

$$
r_{i,d,h}
=
g_i
+ a_{i,d}
+ c_{i,h}
+ u_{i,d,h}
+ \varepsilon_{i,d,h}
$$

各成分の意味は以下である。

| 成分 | 意味 |
|---|---|
| $g_i$ | 店舗・商品系列に固有のズレ |
| $a_{i,d}$ | 日単位のズレ |
| $c_{i,h}$ | 時間帯単位のズレ |
| $u_{i,d,h}$ | 日と時間帯の組み合わせによるズレ |
| $\varepsilon_{i,d,h}$ | ノイズ |

提案モデルでは、予測残差も同じ形で出す。

$$
\hat r_{i,d,h}
=
\hat g_i
+ \hat a_{i,d}
+ \hat c_{i,h}
+ \hat u_{i,d,h}
$$

これにより、従来の

$$
\hat r = f(z_{\mathrm{global}},z_{\mathrm{day}},z_{\mathrm{hour}},z_{\mathrm{interaction}})
$$

のように decoder 内で自由に混ざる設計を避ける。

## 論文としての主張

本研究の主張は、次の形に収束させる。

> 小売需要では、単純な基準値で説明できる部分が大きい。
> 本研究では、売上そのものではなく、基準値からのズレを系列・日・時間帯・日×時間帯の 4 成分として出力する。
> 各成分に平均ゼロ制約を入れることで、同じズレを複数の成分が重複して説明しないようにし、補正量を読める形にする。

重要なのは、**latent 自体の完全な識別を保証するのではなく、出力された残差成分の意味を保証する**ことである。

## これまでの実験との関係

| 実験 | 得られた知見 | proposal への接続 |
|---|---|---|
| 2-Exp-1 | same-hour baseline が FreshRetailNet で強い | 売上全体ではなく残差に集中する |
| 2-Exp-2〜6 | 残差再構成と `b + r_hat` 補正を試した | 単純な residual AE では補正が弱い |
| 2-Exp-7 | swap 正則化あり/なしを比較 | swap だけでは分離保証が弱い |
| 2-Exp-8 | structured synthetic では仮説が成立 | 構造があればモデルは動く |
| 2-Exp-9 | FreshRetailNet subset では baseline を超えにくい | 実データでは残差構造が弱い |
| 2-Exp-10 | factor subset と ablation を追加 | day は少し見えるが hour/interaction は弱い |
| 2-Exp-11〜15 | 出力成分分解と follow-up 実験を追加 | latent 分離より output 分解へ主張を移す |
| 2-Exp-16 | residual target sensitivity を確認 | FreshRetailNet では target 設計が成否を左右する |
| 2-Exp-17 | `series_mean` residual の hour 成分を検証 | hour 構造が残る residual では提案法が意味を持つ |
| 2-Exp-18〜19 | calibration と bias 制約を検証 | 予測補正、bias、高残差改善の trade-off を整理 |
| 2-Exp-20 | paired bootstrap による統計確認 | `series_mean_all` の改善が seed 依存ではないことを確認 |
| 2-Exp-21 | hour profile と heatmap 可視化 | 成功例と限界例を図で示せる |
| 2-Exp-22 | synthetic difficulty final | 成分回復、centering、interaction の必要性を主表にできる |
| 2-Exp-23 | paper table aggregation | 論文用の主要表を再生成可能に固定 |
| 2-Exp-26 | global/local から 4 成分 split への橋渡し | 元論文からの論理の飛躍を減らす |
| 2-Exp-27 | direct/residual bridge | residual target は有効だが latent split だけでは不十分 |
| 2-Exp-28 | latent split vs output decomposition | output decomposition + centering を主提案に置く根拠 |

## ここまでで分かったこと

### 1. Synthetic では成分分解が成立する

2-Exp-22 では、`output_decomp_centered` が synthetic の `base` 条件で次の結果を示した。

- global corr: 0.9995
- day corr: 0.9979
- hour corr: 0.9991
- interaction corr: 0.9654

また、centering なしの `output_decomp_no_center` は residual MAE だけなら極端に悪くないが、global corr が負になり、interaction corr もほぼ崩れた。

これは、予測値を当てることと、成分として読めることが別であることを示している。

### 2. Interaction component は必要な条件で効く

`high_interaction` では、`output_decomp_centered_no_interaction` が悪化し、`output_decomp_centered` の interaction corr は 0.9890 まで上がった。

一方、`low_interaction` では interaction ablation delta が小さくなった。

したがって、interaction component は常に必要というより、interaction が十分に存在する条件で必要になる。

### 3. FreshRetailNet では target 設計が重要である

2-Exp-23 の集約表では、`series_mean_all` と `same_hour_recent_mean_d7_all` の差が明確だった。

`series_mean_all`:

- baseline MAE: 0.0697
- `bias_constrained_001` corrected MAE: 0.0501
- calibrated high residual top10 corrected MAE: 0.1914
- hour profile corr: 0.9938

`same_hour_recent_mean_d7_all`:

- baseline MAE: 0.0580
- `bias_constrained_001` corrected MAE: 0.0565
- calibrated corrected MAE: 0.0584
- hour profile corr: -0.8766

つまり、残差に hour 構造が残る target では提案法が効くが、same-hour baseline が hour 構造を消した後では効果は小さい。

### 4. 統計的には `series_mean_all` が主成功例である

2-Exp-20 / 2-Exp-23 では、`series_mean_all` の改善が 5 seed で一貫した。

- `bias_constrained_001` の `corrected_cell_mae` は baseline より `-0.0196` 改善し、CI は `[-0.0201, -0.0189]`。
- `calibrated_high_residual_top10_corrected_mae` は `-0.0874` 改善し、CI は `[-0.0905, -0.0842]`。

この結果により、FreshRetailNet での主張は「全ての baseline を置き換える」ではなく、「残差構造が残る target で補正と解釈が有効」という形にする。

## まだ不十分なこと

### 1. FreshRetailNet 全体で常に勝つ主張はできない

`same_hour_recent_mean_d7_all` では改善が小さく、calibration 後に全体 MAE が悪化する条件もある。

したがって、論文では次のような強い主張は避ける。

```text
提案法は FreshRetailNet で常に強い baseline を上回る。
```

代わりに、次を主張する。

```text
提案法は、baseline 後の残差に day/hour/interaction 構造が残る場合に、補正と解釈の両方で有効である。
```

### 2. 実データの interaction 成分は主張しすぎない

synthetic では interaction の必要性を確認できたが、FreshRetailNet では interaction component の強い成功例はまだ主張しにくい。

本文では hour component を実データの主成功例にし、interaction は synthetic と appendix 中心に扱う。

### 3. 数理保証は出力成分に限定される

本研究で保証できるのは、latent の完全な識別ではない。

保証できるのは、出力成分が平均ゼロ制約と主効果除去制約を満たすことで、系列・日・時間帯・日×時間帯の担当範囲を持つ成分として検証可能になることである。

### 4. 本文用の表と appendix 用の表を分ける必要がある

2-Exp-23 により表は再生成可能になったが、`statistical_validation` は 56 行あり、そのまま本文に載せるには大きい。

次は追加実験ではなく、本文用の小さい表と appendix 用の詳細表に分ける作業が必要である。

## 論文 1 本にするための必要条件

論文として成立させるには、最低限以下が必要である。

1. **数理的整理**
   - 残差成分の担当範囲が一意に定まる条件を示す。
   - 提案モデルの平均ゼロ制約がその担当範囲に対応することを示す。

2. **合成データでの検証**
   - 真の $g,a,c,u$ が分かるデータで、成分回復を評価する。
   - 現行モデルより提案モデルの方が成分を正しく回復することを示す。

3. **実データでの検証**
   - FreshRetailNet で、予測補正だけでなく、成分別 ablation、leakage、subset 別挙動を見る。
   - 全体平均で勝たなくても、残差構造が強い subset で意味が出るかを見る。

4. **比較対象**
   - same-hour baseline
   - ANOVA 的な非学習分解
   - 現行 latent concat model
   - single local model
   - output decomposition model
   - output decomposition + constraints

5. **再現性**
   - 3〜5 seeds
   - subset 別評価
   - ablation
   - paired bootstrap または seed 平均・標準偏差

## ドキュメント構成

- [related_work_and_improvement.md](related_work_and_improvement.md): 周辺研究の整理、現状課題、改良方針
- [formulation.md](formulation.md): 基準値 $b$、残差 $y-b$、平均ゼロ制約の定式化
- [paper_direction.md](paper_direction.md): 6 月末原稿に向けた主張の再整理
- [theory.md](theory.md): 数理的な保証、識別可能性、収束の考え方
- [experiments.md](experiments.md): 論文レベルに必要な実験一覧
- [schedule.md](schedule.md): 実験と執筆のスケジュール
- [risks.md](risks.md): 失敗パターンと縮退案

## 現時点の結論

この研究は、単に元論文を FreshRetailNet に適用する方向では弱い。

一方で、今回の失敗分析を利用して、

> 元論文の暗黙的な global/local 分離は、実データでは local の中に日・時間帯・日×時間帯が混ざりやすい。
> そこで、残差補正を 4 つの出力成分として直接出し、平均ゼロ制約で担当範囲を固定することで、解釈可能性と検証可能性を高める。

という形にすれば、十分に論文の核になる。

現時点の論文の核は次である。

```text
Synthetic では、成分が存在する条件で output decomposition が成分を回復できる。
FreshRetailNet では、残差に hour 構造が残る target で予測補正と high residual 改善が確認できる。
一方、強い same-hour baseline 後の残差では構造が薄く、改善は小さい。
```
