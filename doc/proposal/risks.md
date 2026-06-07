# Risks: 失敗パターンと縮退案

## 1. 主要リスク

### R1: FreshRetailNet で補正性能が改善しない

これまでの実験でも、same-hour baseline は非常に強かった。
そのため、$b+\hat r$ が $b$ を全体平均で超えない可能性は高い。

2-Exp-23 時点では、このリスクは半分現実化し、半分解消した。

- `series_mean_all` では baseline 改善が明確に出た。
- `same_hour_recent_mean_d7_all` では改善が小さく、calibration 後に全体 MAE が悪化する条件もある。

### 対応

主張を「全ての baseline を上回る予測器」ではなく、「残差構造が残る条件で有効な補正・分解」に寄せる。

成功条件を、

```text
全体 MAE 改善
```

ではなく、

```text
component recovery
leakage reduction
factor subset ablation
high residual subset 改善
residual target ごとの成立条件
```

に置く。

## 2. R2: hour / interaction が分離しない

2-Exp-10 では、day は少し見えたが、hour と interaction は弱かった。

2-Exp-17〜23 で、hour については `series_mean_all` で解消した。一方、interaction については FreshRetailNet で強く主張できるほどの結果はまだない。

### 原因候補

- residual に hour 構造がそもそも弱い。
- same-hour baseline が hour 構造を取り除きすぎている。
- interaction subset の切り方が不十分。
- decoder がまだ成分を混ぜている。

### 対応

- baseline を変える。
  - same-hour ではなく recent mean を使う条件を追加する。
- interaction strength が既知の synthetic でまず確認する。
- FreshRetailNet では interaction 主張を弱める。
- 論文では day/hour/interaction のうち、成立する成分を中心に主張する。

現時点の方針:

```text
実データの主成功例は hour component とする。
interaction component は synthetic での成立条件として示し、FreshRetailNet では limitation として扱う。
```

## 3. R3: leakage suppression で再構成が壊れる

情報漏れを抑えすぎると、必要な情報まで消える可能性がある。

2-Exp-23 時点では、leakage suppression は本文の主軸から外してよい。
主提案は output decomposition + centering であり、leakage suppression は appendix または今後課題に回す。

### 対応

- leakage loss weight を小さく始める。
- まず subgroup leakage だけを抑える。
- reconstruction と leakage の Pareto curve を出す。
- 「leakage を下げつつ、再構成を保つ」範囲を探す。

## 4. R4: synthetic では成功するが real では弱い

これは十分あり得る。

2-Exp-23 時点の結果は、このリスクを「失敗」ではなく「適用条件」として書ける状態である。

- synthetic では成分回復が強い。
- FreshRetailNet では `series_mean_all` で成功する。
- FreshRetailNet の `same_hour_recent_mean_d7_all` では効果が小さい。

### 対応

この場合、論文の主張を以下にする。

```text
暗黙的な latent 分離は、真の成分が存在する場合には動く。
しかし実データでは、成分構造が弱いと分離が不安定になる。
本研究は、その問題に対して出力直交分解という検証可能な設計を提案する。
```

この主張でも修士論文としては成立する。
投稿論文としては、real での改善が `series_mean_all` に限定されるため、主張を過大にしないことが重要である。

## 5. R5: 数理保証が弱く見える

ニューラルネットの latent を完全に識別する保証は難しい。

### 対応

保証対象を latent ではなく、出力成分に限定する。

主張は、

```text
latent が一意に識別される
```

ではなく、

```text
出力成分が ANOVA 的な制約空間に属するため、成分の意味を検証できる
```

にする。

2-Exp-22 の結果により、この方針は実験的にも支持された。

`output_decomp_no_center` は residual MAE だけなら極端に悪くないが、成分 corr が崩れた。したがって、出力制約がなければ「当たるが読めない」状態になることを示せる。

## 6. 縮退案

### Plan A

出力直交分解 + leakage suppression で、synthetic と FreshRetailNet の両方で改善する。

これは最も強い。

現時点では、leakage suppression までは主張しないため Plan A からは少し弱める。

### Plan B

FreshRetailNet の予測補正は弱いが、leakage と ablation が改善する。

この場合は、解釈可能な残差分解として主張する。

現時点の主ラインは Plan B に近いが、`series_mean_all` では予測補正も確認できている。

### Plan C

synthetic では強く、FreshRetailNet では弱い。

この場合は、実データで暗黙的分離が難しいことを示す失敗分析型の修士論文にする。

現時点では Plan C より強い。FreshRetailNet でも target を選べば改善と hour component が出ている。

### Plan D

提案法が既存モデルと大差ない。

この場合は、以下に縮退する。

- FreshRetailNet では same-hour baseline が強すぎることを体系的に示す。
- 残差構造が弱い条件で、表現分離が成立しにくいことを示す。
- synthetic-to-real gap を研究課題として整理する。

現時点では Plan D まで縮退する必要はない。

## 7. 現時点の残リスク

2-Exp-23 後に残っているリスクは次の 4 つである。

| リスク | 現状 | 論文での扱い |
|---|---|---|
| FreshRetailNet 全体で常に勝つわけではない | `same_hour_recent_mean_d7_all` は改善が小さい | 適用条件として説明 |
| real data interaction が弱い | synthetic では強いが real では主張しにくい | synthetic 中心、real は limitation |
| 本文表が多すぎる | 2-Exp-23 の statistical table は 56 行 | 本文用と appendix 用に分ける |
| target selection が恣意的に見える | `series_mean_all` が主成功例 | 2-Exp-16/23 の target sensitivity として説明 |

## 8. 中止判断

以下が 2026-07-31 時点で満たせない場合、投稿論文としての主張は弱くなる。

- synthetic component recovery で P2/P3 が B3 に勝たない。
- leakage suppression が leakage を下げない。
- FreshRetailNet のどの subset でも意味のある ablation が出ない。

この場合は、修士論文向けの失敗分析・設計提案に切り替える。

2-Exp-23 時点では、synthetic component recovery と FreshRetailNet の `series_mean_all` 改善は満たしているため、中止判断には該当しない。
