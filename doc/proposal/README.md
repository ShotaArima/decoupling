# Proposal: 残差直交分解に基づく小売需要表現学習

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

これまでの結果から、FreshRetailNet では $b$ が非常に強く、単純に $r$ を学習しても、平均的な予測補正では $b$ を超えにくいことが分かった。

一方で、合成データでは、残差に明確な構造がある場合、`global/day/hour/interaction` の考え方は成立した。

したがって次の研究の中心は、単に latent を分けることではなく、**残差の出力成分そのものを数理的に分けること**に置く。

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

> 小売需要の残差は、基準値で説明できない小さなズレであるため、単純な潜在分離だけでは意味のある分離が安定しない。
> 本研究では、残差を global/day/hour/interaction の直交成分として出力空間で分解し、各成分に平均ゼロ制約と主効果除去制約を入れる。
> これにより、暗黙的な latent 分離よりも、解釈可能で検証可能な残差分解が得られる。

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

## 論文 1 本にするための必要条件

論文として成立させるには、最低限以下が必要である。

1. **数理的整理**
   - 残差の直交分解が一意に定まる条件を示す。
   - 提案モデルの出力制約がその分解に対応することを示す。

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

- [theory.md](theory.md): 数理的な保証、識別可能性、収束の考え方
- [experiments.md](experiments.md): 論文レベルに必要な実験一覧
- [schedule.md](schedule.md): 実験と執筆のスケジュール
- [risks.md](risks.md): 失敗パターンと縮退案

## 現時点の結論

この研究は、単に元論文を FreshRetailNet に適用する方向では弱い。

一方で、今回の失敗分析を利用して、

> 元論文の暗黙的な分離は実データでは不安定である。
> そこで、出力空間に直交分解を入れることで、解釈可能性と検証可能性を高める。

という形にすれば、十分に論文の核になる。

