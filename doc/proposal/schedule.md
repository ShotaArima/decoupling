# Schedule: 論文化までの残り実験計画

## 現在地

更新日: 2026-06-07

当初の 12 週間計画のうち、実装と探索の多くは前倒しで進んでいる。

現時点で分かっていること:

- synthetic では、成分を持つ残差に対して output decomposition が有効に働く。
- FreshRetailNet では、`same_hour_recent_mean` のような強い baseline の残差は構造が薄い。
- `series_mean` residual では hour 構造が残り、hour component の寄与が安定して大きい。
- 2-Exp-17〜19 で、FreshRetailNet でも `series_mean_all` とカテゴリ集約条件では baseline より低い MAE が出た。
- bias 制約つき calibration は bias を抑え、高残差上位 10% の改善を強める一方、全体 MAE は `mae_grid_reference` より悪化する。

したがって、論文の主張は次に寄せるのが現実的である。

```text
提案する残差分解は、強い baseline を置き換える手法ではなく、
baseline 後の残差に残る day/hour 構造を分解し、
条件が合う場合に予測補正と外れケース改善を与える。
```

## あと必要な実験数

最低限は 4 本。

| ID | 目的 | 必須度 |
| --- | --- | --- |
| `2-Exp-20` | 2-Exp-19 の seed-level paired bootstrap | 必須 |
| `2-Exp-21` | 成功例・失敗例の heatmap / residual profile 可視化 | 必須 |
| `2-Exp-22` | synthetic difficulty の最終表を固定 | 必須 |
| `2-Exp-23` | 論文用 final table を 1 つの summary に統合 | 必須 |

余裕があれば追加で 2 本。

| ID | 目的 | 必須度 |
| --- | --- | --- |
| `2-Exp-24` | FreshRetailNet subset threshold の軽い感度確認 | 任意 |
| `2-Exp-25` | ablation / leakage probe の appendix 用補強 | 任意 |

つまり、論文 1 本の骨格には「あと 4 本」、査読耐性を上げるなら「あと 6 本」が目安。

## Revised Week 1: 統計検証と採用モデル決定

期間:

```text
2026-06-07 〜 2026-06-13
```

### 実験

- `2-Exp-20`: 2-Exp-19 の seed-level paired bootstrap

### 完了条件

- `series_mean_all` で baseline 改善の CI が確認できる。
- `bias_constrained_001` を主モデルにするか、`mae_grid_reference` を主モデルにするかを決める。
- 全体 MAE、bias、高残差 top10 の trade-off を表にできる。

## Revised Week 2: 可視化と成功・失敗例

期間:

```text
2026-06-14 〜 2026-06-20
```

### 実験

- `2-Exp-21`: residual heatmap / hour profile / component profile

### 完了条件

- 成功例 3 件、失敗例 3 件を選べる。
- `series_mean_all` で hour component が residual hour profile と対応する図がある。
- `same_hour_recent_mean_d7_all` で残差構造が薄いことを示す図がある。

## Revised Week 3: Synthetic 最終表

期間:

```text
2026-06-21 〜 2026-06-27
```

### 実験

- `2-Exp-22`: synthetic difficulty final

### 完了条件

- noise / interaction / missing の条件別に、どこで成分分解が成立し、どこで失敗するかを表にできる。
- true component がある synthetic で、同定可能性の主張を支える。

## Revised Week 4: Final Table 統合

期間:

```text
2026-06-28 〜 2026-07-04
```

### 実験

- `2-Exp-23`: paper table aggregation

### 完了条件

- Synthetic main table
- FreshRetailNet correction table
- Calibration trade-off table
- Limitation table

を 1 つの出力ディレクトリに固定する。

## Revised Week 5: 任意の補強

期間:

```text
2026-07-05 〜 2026-07-11
```

### 実験

- `2-Exp-24`: subset threshold sensitivity
- `2-Exp-25`: leakage / ablation appendix

### 完了条件

- 査読で聞かれやすい「subset を選んだから良いだけではないか」に答えられる。
- 表現分離について appendix に補足表を出せる。

## Revised Week 6-7: 執筆

期間:

```text
2026-07-12 〜 2026-07-25
```

### 作業

- Introduction
- Method
- Theory
- Experiments
- Discussion
- Limitations

を書く。

### 完了条件

- 8 ページ相当の初稿がある。
- 主要図表が本文から参照されている。
- FreshRetailNet の主張が過大ではなく、synthetic と real data の役割分担が明確である。

## Revised Week 8: 投稿判断

期間:

```text
2026-07-26 〜 2026-08-01
```

### 判断基準

#### Case A

2-Exp-20 で baseline 改善の CI が明確に 0 未満。

主張:

```text
残差分解は解釈可能性と予測補正の両方に寄与する。
```

#### Case B

MAE の CI は弱いが、hour component / high residual / synthetic が強い。

主張:

```text
残差分解は、強い baseline 下の残差構造を解釈し、
一部条件で予測補正にも寄与する。
```

#### Case C

FreshRetailNet の統計的改善が弱い。

主張:

```text
残差分解の成立条件を synthetic と実データで整理し、
強い baseline 後の residual learning の限界を示す。
```

現時点では Case B が最も現実的で、2-Exp-20 の結果次第で Case A に寄せられる。
