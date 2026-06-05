# Schedule: 実験スケジュール

## 前提

開始日を 2026-06-06 とする。

目的は、まず 8 週間で論文として成立するかを判断できる材料を揃えることである。
その後、追加 4 週間で図表・統計・執筆に進む。

全体として、12 週間で「投稿可能な原稿の骨格」まで持っていく計画にする。

## Week 1: 設計固定と実装準備

期間:

```text
2026-06-06 〜 2026-06-12
```

### 目的

提案法の設計を固定し、実装対象を明確にする。

### 作業

- 出力分解モデルの仕様を確定する。
- $\hat g,\hat a,\hat c,\hat u$ の出力形状を決める。
- centering constraint を実装する設計を決める。
- synthetic data に真の $g,a,c,u$ を保存する。
- 既存の residual experiment runner に component metrics を追加する設計を決める。

### 完了条件

- `OutputDecompositionResidualModel` の設計メモがある。
- 実装すべき metrics がリスト化されている。
- synthetic component recovery の評価式が確定している。

## Week 2: Output Decomposition Model 実装

期間:

```text
2026-06-13 〜 2026-06-19
```

### 目的

提案モデル P1/P2 を動かす。

### 作業

- P1: output decomposition model を実装する。
- P2: centering constraints を追加する。
- residual runner から実行できるようにする。
- smoke config を作る。
- `g_hat/a_hat/c_hat/u_hat` を保存する。

### 完了条件

- smoke synthetic が通る。
- `summary.json` に component metrics が出る。
- `g_hat/a_hat/c_hat/u_hat` が保存される。

## Week 3: Controlled Synthetic 実験

期間:

```text
2026-06-20 〜 2026-06-26
```

### 目的

合成データで、提案法が成分を回復できるかを見る。

### 作業

- noise level を 3 段階で振る。
- interaction strength を 3 段階で振る。
- missing rate を 2 段階で振る。
- B2/B3/P1/P2 を比較する。
- 3 seeds で回す。

### 完了条件

- component recovery table が作れる。
- P2 が B3 より $\mathrm{MAE}_g,\mathrm{MAE}_a,\mathrm{MAE}_c,\mathrm{MAE}_u$ のいずれか、できれば全体平均で勝つ。
- interaction strength が高い条件で $\hat u$ が意味を持つ。

## Week 4: Leakage Suppression 実装

期間:

```text
2026-06-27 〜 2026-07-03
```

### 目的

P3 を実装する。

### 作業

- leakage probe を adversarial loss 化する。
- まずは subgroup leakage を対象にする。
- 次に discount / weekday / hour leakage を必要に応じて追加する。
- P2 と P3 を synthetic で比較する。

### 完了条件

- P3 が動く。
- leakage が P2 より下がる。
- reconstruction が大きく壊れない。

## Week 5: FreshRetailNet 基本評価

期間:

```text
2026-07-04 〜 2026-07-10
```

### 目的

FreshRetailNet で P2/P3 を評価する。

### 作業

- Full subset
- active subset
- high residual subset
- day/hour/interaction structured subset

を評価する。

比較対象は、

- B0: same-hour baseline
- B1: ANOVA direct
- B3: latent concat
- P2
- P3

とする。

### 完了条件

- `aggregate.csv` が subset x model で出る。
- corrected MAE と baseline MAE の比較ができる。
- ablation delta が出る。
- leakage probe が出る。

## Week 6: FreshRetailNet 追加評価と失敗分析

期間:

```text
2026-07-11 〜 2026-07-17
```

### 目的

FreshRetailNet で主張できる範囲を決める。

### 作業

- high residual top10 の改善を見る。
- zero / nonzero 系列を分ける。
- stockout 多い系列を除いた評価を行う。
- subset filter の閾値を軽く振る。
- 失敗例を residual heatmap で確認する。

### 完了条件

- 「予測補正で主張できるか」または「解釈可能な残差分解に主張を絞るか」を判断する。
- 失敗例と成功例が 3 つ以上ある。

## Week 7: Robustness

期間:

```text
2026-07-18 〜 2026-07-24
```

### 目的

seed 依存を確認する。

### 作業

- 主要条件を 5 seeds で回す。
- mean/std を出す。
- paired bootstrap を実装する。
- synthetic と FreshRetailNet の主要表を固定する。

### 完了条件

- 最終候補モデルが 1 つに絞れている。
- 主要結果が 5 seeds で出ている。
- 統計的に弱い箇所が把握できている。

## Week 8: 論文判断

期間:

```text
2026-07-25 〜 2026-07-31
```

### 目的

論文の主張を確定する。

### 判断分岐

#### Case A: FreshRetailNet で補正も改善

主張:

```text
提案する残差直交分解は、解釈可能性と予測補正の両方を改善する。
```

#### Case B: 補正は弱いが分離性は改善

主張:

```text
提案する残差直交分解は、強い baseline 下でも、残差構造を解釈可能に分解する。
```

#### Case C: synthetic では成功、real では弱い

主張:

```text
暗黙的な latent 分離の限界を実証し、実データで必要な条件を整理する。
```

Case A または B なら論文として前進する。
Case C のみなら、投稿論文ではなく修士論文の失敗分析章として扱う。

## Week 9-10: 図表作成

期間:

```text
2026-08-01 〜 2026-08-14
```

### 作業

- Table 1: synthetic component recovery
- Table 2: FreshRetailNet correction
- Table 3: factor subset ablation
- Table 4: leakage probe
- Figure 1: model overview
- Figure 2: component heatmap
- Figure 3: failure/success examples

### 完了条件

- 論文本文に貼れる図表が揃う。
- captions が書ける。

## Week 11: 初稿

期間:

```text
2026-08-15 〜 2026-08-21
```

### 作業

- Introduction
- Related Work
- Method
- Theory
- Experiments
- Discussion

の初稿を書く。

### 完了条件

- 8 ページ相当の初稿がある。
- 主要図表がすべて参照されている。

## Week 12: 修正と提出判断

期間:

```text
2026-08-22 〜 2026-08-28
```

### 作業

- 主張の強さを調整する。
- 追加実験が必要か判断する。
- Appendix に実験詳細をまとめる。
- 再現手順を整理する。

### 完了条件

- 投稿可能性の判断ができる。
- 修士論文の中核章として使える状態になっている。

## 最短で確認すべき実験

時間がない場合は、以下だけを優先する。

1. controlled synthetic で P2 が B3 より成分回復で勝つ。
2. FreshRetailNet high residual subset で P2/P3 を評価する。
3. P2/P3 の leakage が B3 より下がる。
4. ablation delta が subset の意味と合う。

この 4 つが揃えば、論文の核は立つ。

