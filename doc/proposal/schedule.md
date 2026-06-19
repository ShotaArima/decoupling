# Schedule: 論文化までの残り実験計画

## 現在地

更新日: 2026-06-19

6 月 9 日のゼミコメントを受けて、以降の優先順位を変更する。

これ以上の実験追加よりも、まず次の 3 点を原稿として伝わる形にする。

1. なぜ売上 $y$ そのものではなく、基準値からのズレ $r=y-b$ を扱うのか。
2. 従来の global/local 分離から、系列・日・時間帯・日×時間帯の 4 成分へ拡張することが、誰にとってどのような意味を持つのか。
3. 平均ゼロ制約によって、各成分の担当範囲をどのように固定するのか。

当初の 12 週間計画のうち、実装と探索の多くは前倒しで進んでいる。

現時点で分かっていること:

- synthetic では、成分を持つ残差に対して output decomposition が有効に働く。
- FreshRetailNet では、`same_hour_recent_mean` のような強い baseline の残差は構造が薄い。
- `series_mean` residual では hour 構造が残り、hour component の寄与が安定して大きい。
- 2-Exp-17〜19 で、FreshRetailNet でも `series_mean_all` とカテゴリ集約条件では baseline より低い MAE が出た。
- 2-Exp-20 で、`series_mean_all` の baseline 改善は 5 seed の paired bootstrap でも 0 未満になった。
- 2-Exp-21 で、`series_mean_all` は hour component が強く、`same_hour_recent_mean_d7_all` は hour component が薄いことを確認した。
- 2-Exp-22 で、synthetic では `output_decomp_centered` が true component を高い相関で回復し、centering と interaction component の必要性も確認できた。
- 2-Exp-23 で、synthetic / FreshRetailNet / statistical validation の論文用表を CSV/Markdown として再生成可能にした。
- 2-Exp-24 で、FreshRetailNet の `series_mean` residual は 12000 train 系列まで増やしても改善が保たれ、hour component の対応も安定した。
- 2-Exp-24 では、`same_hour_recent_mean_d7` residual も MAE は少し改善したが、hour component の対応は負相関になり、解釈可能な分解としては不安定だった。
- 2-Exp-25 で、`series_mean` residual の改善は先頭系列ブロックだけでなく、開始位置をずらした 3 つの 6000 系列ブロックでも再現した。
- 2-Exp-25 でも、`same_hour_recent_mean_d7` residual は改善幅が小さく、hour component の対応は負相関であり、強い基準値下では解釈可能な残差構造が残りにくいことを確認した。
- 2-Exp-26 で、元論文に近い `global/local` 2 成分から `global/day/hour/(interaction)` へ拡張する比較を行った。`day/hour` split は direct target でも小さな改善を示したが、interaction は安定した改善にはならなかった。
- 2-Exp-27 で、同じ比較を `series_mean` residual に対して行った。residual 学習自体は baseline を改善したが、4 成分 latent が `global/local` residual を常に上回るわけではなかった。
- 周辺研究レビューを踏まえると、本研究は「latent を細かく分ける研究」ではなく、強い baseline 後に残る構造を output-level decomposition と centering constraints で説明する研究として固定するのがよい。
- bias 制約つき calibration は bias を抑え、高残差上位 10% の改善を強める一方、全体 MAE は `mae_grid_reference` より悪化する。

したがって、論文の主張は次に寄せるのが現実的である。

```text
提案する残差分解は、強い baseline を置き換える手法ではない。
元論文の global/local 表現分離を出発点にしつつ、
baseline 後の残差に残る series/day/hour/interaction 構造を
出力空間で制約付きに分解し、
条件が合う場合に予測補正、外れケース改善、成分解釈を与える。
```

現時点で新しい大規模実験を増やす優先度は高くない。
優先すべき作業は、研究の方向性を固定し、その方向に合わせて定式化、関連研究、実験結果の読み方を揃えることである。

## あと必要な実験数

最低限の実験は完了。

| ID | 目的 | 必須度 |
| --- | --- | --- |
| `2-Exp-23` | 論文用 final table を 1 つの summary に統合 | 完了 |

今後の展望として、査読耐性と応用可能性を上げるための追加実験を行う。

| ID | 目的 | 必須度 |
| --- | --- | --- |
| `2-Exp-24` | FreshRetailNet の系列数感度確認 | 完了 |
| `2-Exp-25` | FreshRetailNet の系列ブロック頑健性確認 | 完了 |
| `2-Exp-26` | 元論文 `global/local` から 4 成分分割への橋渡し | 完了 |
| `2-Exp-27` | direct target と residual target の最小比較 | 完了 |
| `2-Exp-28` | latent split と output decomposition の直接比較 | 追加実験 |

つまり、論文 1 本の骨格に必要な実験は揃っている。
2-Exp-24 と 2-Exp-25 により、「系列数を増やしても保たれるか」「系列ブロックを変えても保たれるか」は補強できた。
2-Exp-26 と 2-Exp-27 により、元論文に近い global/local 分離から残差の output decomposition へ進む論理も整理できた。

ここから先の追加実験は、主張の中心を作るためではなく、限界として残っている「基準値選択に依存する」「実データ interaction が弱い」「latent split だけでは成分解釈を保証しない」という点を補足するために行う。
その中で 2-Exp-28 は、latent split と output decomposition を同じ residual target 上で直接比較し、proposal の主張を閉じるための最小追加実験として扱う。

## 直近で自分がやること

結論として、直近の主作業は **方向性の固定と、それに向かうための論理補強** である。
定式化と実験は必要だが、どちらも新しい主張を増やすためではなく、固定した方向性を支える範囲に絞る。

| 作業 | 優先度 | 目的 | 完了条件 |
|---|---:|---|---|
| 方向性の固定 | 最優先 | 論文の中心を residual output decomposition に固定する | Introduction / Method / Experiments で同じ主張になっている |
| 定式化 | 高 | `y=b+r`, `r=g+a+c+u+eps`, centering constraints, correction `b+\hat r` を明確に書く | Method 節だけ読めば、何を分解しているか分かる |
| 論理補強 | 高 | 周辺研究と Exp-26/27 を使い、なぜ latent split だけではなく output decomposition なのか説明する | Related Work / Discussion に橋渡しの説明がある |
| 実験整理 | 高 | 既存実験から本文表と appendix 表を選ぶ | Main table 3 個、main figure 2〜3 個が決まっている |
| 追加実験 | 低 | 指導教員から不足を指摘された場合の補強 | 具体的な弱点が出た場合だけ実施 |

### 定式化でやること

Method 節では、複雑な理論を増やすより、以下を一貫して書く。

```text
y_{i,d,h} = b_{i,d,h} + r_{i,d,h}
r_{i,d,h} = g_i + a_{i,d} + c_{i,h} + u_{i,d,h} + eps_{i,d,h}
hat y_{i,d,h} = b_{i,d,h} + hat r_{i,d,h}
```

加えて、day/hour/interaction の centering constraints を書く。
ここでの主張は「潜在変数が完全に同定される」ではなく、「出力成分に構造制約を置くことで、補正値を series/day/hour/interaction として読めるようにする」である。

### 実験でやること

現時点では、新規実験を増やすより、既存結果を論文の問いに対応づける。

| 問い | 使う実験 |
|---|---|
| 成分が存在すれば回復できるか | 2-Exp-22 |
| 実データで baseline を補正できるか | 2-Exp-17〜20 |
| 可視的に hour 構造が残っているか | 2-Exp-21 |
| 系列数・系列ブロックを変えても成立するか | 2-Exp-24, 2-Exp-25 |
| 元論文の global/local から飛躍していないか | 2-Exp-26 |
| residual にすれば常に 4 成分が良いのか | 2-Exp-27 |
| latent split より output decomposition を主提案にする根拠はあるか | 2-Exp-28 |

2-Exp-27 の結果から、「residual なら常に 4 成分 latent が良い」とは書かない。
書くべきことは、「residual target は baseline 改善に有効だが、成分解釈を安定させるには latent split ではなく output decomposition と centering が必要である」である。

### 論理補強でやること

周辺研究は次の流れで使う。

| 周辺研究の流れ | 本研究での受け方 |
|---|---|
| FHVAE / DSVAE / C-DSVAE | static/dynamic や global/local の表現分離は重要だが、latent だけでは成分の意味が曖昧になり得る |
| TS2Vec / TS-TCC / TF-C | 時系列表現学習は有効だが、需要予測の補正値を day/hour/interaction として読む目的とは少し違う |
| Deep Factors / DeepGLO / CoST | global/local forecasting や seasonal decomposition と接続できる |
| Shapelet / ROCKET / anomaly 系 | 外れケース、高残差 subset、運用支援の評価につなげられる |

この整理により、提案手法は「既存の表現学習を否定する」のではなく、「小売需要補正では、表現分離を residual output decomposition として制約付きに実装する必要がある」と位置づける。

## 6 月末原稿締切に向けた作業順

締切目標:

```text
2026-06-30
```

### 2026-06-09 〜 2026-06-11: 定式化の固定

目的:

- `formulation.md` を中心に、$b$、$r=y-b$、4 成分出力、平均ゼロ制約を説明できる状態にする。
- 「直交分解」という語を本文の中心から外し、「残差成分モデル」「平均ゼロ制約」「担当範囲」という語に寄せる。
- 想定質問「なぜ $y$ をそのまま予測しないのか」に対する 1 分回答と本文回答を用意する。

完了条件:

- Problem formulation の節にそのまま移せる数式と文章がある。
- `b` が真の成分ではなく、既存の基準値、比較対象、診断軸であると説明できる。
- 平均ゼロ制約が「成分を小さくする制約」ではなく「成分の担当範囲を固定する制約」だと説明できる。

### 2026-06-12 〜 2026-06-14: Introduction と対象読者の整理

目的:

- 誰に対して、どのようなモデルを作るのかを明確にする。
- 予測精度だけでなく、基準値からのズレを説明できることの実務上の価値を書く。

本文に入れる利用者:

| 利用者 | 恩恵 |
|---|---|
| 発注・補充担当者 | 基準値より外れやすい時間帯を把握できる |
| 店舗運営者 | 店舗・商品ごとの基準値のズレを見られる |
| 販促担当者 | 日と時間帯の組み合わせで外れる条件を探せる |
| 需要分析者 | どの基準値を使うと残差に構造が残るか診断できる |

完了条件:

- Abstract / Introduction に、残差を見る理由と利用者の利益が入っている。
- 「高性能な予測器を作る」だけでなく、「基準値の弱い部分を説明する」研究として読める。

### 2026-06-15 〜 2026-06-17: Method の本文化

目的:

- 4 成分出力モデルを、従来の global/local 分離から自然に導入する。
- Encoder / Decoder / 損失関数を、本文と appendix に分けて整理する。

本文に入れる最小構成:

$$
r_{i,d,h}=y_{i,d,h}-b_{i,d,h}
$$

$$
\hat r_{i,d,h}
=
\hat g_i
+
\hat a_{i,d}
+
\hat c_{i,h}
+
\hat u_{i,d,h}
$$

$$
\hat y_{i,d,h}=b_{i,d,h}+\hat r_{i,d,h}
$$

完了条件:

- Encoder と Decoder の入出力が説明されている。
- 損失関数が、再構成、平均ゼロ制約、bias 制約、calibration に分かれて説明されている。
- 平均ゼロ制約の直感が本文に入り、詳細式は appendix に逃がせる。

### 2026-06-18 〜 2026-06-20: 図表の選定

目的:

- 実験結果を増やすのではなく、主張に必要な表と図だけを選ぶ。

本文候補:

| 種類 | 内容 |
|---|---|
| Table 1 | synthetic component recovery |
| Table 2 | FreshRetailNet correction |
| Table 3 | statistical validation の要約 |
| Figure 1 | 提案モデルの全体図 |
| Figure 2 | series_mean 成功例の heatmap / hour profile |
| Figure 3 | same-hour baseline の限界例 |

完了条件:

- 本文図表と appendix 図表が分かれている。
- FreshRetailNet は「主成功例」と「限界例」を並べて説明できる。

### 2026-06-21 〜 2026-06-23: 原稿 v0.1

目的:

- Introduction、Problem formulation、Method、Experiments、Discussion を一通りつなげる。

完了条件:

- 行間が大きい箇所に TODO が残っていても、論文全体の流れが読める。
- 「なぜ $b$ を置くのか」「なぜ 4 成分なのか」「何が言えて何が言えないのか」が本文中に出ている。

### 2026-06-24 〜 2026-06-26: 説明不足の修正

目的:

- ゼミコメントで出た「数式と言いたいことのつながりが見えない」を潰す。

確認項目:

- 数式の前に、その式で何を言いたいのかが書かれている。
- 数式の後に、その式が利用者にとって何を意味するかが書かれている。
- 「標準化だけではない」理由が説明されている。
- 「平均ゼロ制約がどう効くか」が説明されている。

### 2026-06-27 〜 2026-06-29: 最終整形

目的:

- 表記揺れ、図表番号、appendix、参考文献、実験条件を整える。

完了条件:

- 本文から「直交分解」という語が主張の中心として出ていない。
- FreshRetailNet-50K で全系列・全交差検証を行ったような誤解がない。
- subset、seed、baseline、評価指標が明記されている。

### 2026-06-30: 提出版

目的:

- 追加実験ではなく、原稿として提出できる形に固定する。

この日までに追加実験が必要になるのは、本文の論理が明確に欠けた場合に限る。

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

状態:

```text
完了。2-Exp-20 で series_mean_all の baseline 改善 CI は 0 未満。
```

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

状態:

```text
完了。visualization CSV からの図生成は scripts/plot_2_exp_21_heatmaps.py で対応済み。
```

## Revised Week 3: Synthetic 最終表

期間:

```text
2026-06-08 〜 2026-06-14
```

### 実験

- `2-Exp-22`: synthetic difficulty final

### 完了条件

- noise / interaction / missing の条件別に、どこで成分分解が成立し、どこで失敗するかを表にできる。
- true component がある synthetic で、同定可能性の主張を支える。

状態:

```text
完了。output_decomp_centered は base で global/day/hour をほぼ完全に回復し、interaction も高く回復した。small_sample と high_noise では成分回復が落ち、失敗条件も表にできた。
```

## Revised Week 4: Final Table 統合

期間:

```text
2026-06-08 〜 2026-06-14
```

### 実験

- `2-Exp-23`: paper table aggregation

### 完了条件

- Synthetic main table
- FreshRetailNet correction table
- Calibration trade-off table
- Limitation table

を 1 つの出力ディレクトリに固定する。

状態:

```text
完了。2-Exp-23 で synthetic_component_recovery、freshretail_correction、statistical_validation を生成した。
```

主な読み取り:

- Synthetic は成分回復と centering の必要性を支える主表として使える。
- FreshRetailNet は `series_mean_all` を主成功例、`same_hour_recent_mean_d7_all` を限界例として使える。
- `statistical_validation` は本文には大きすぎるため、本文用の小表と appendix 用の詳細表に分ける。

## Revised Week 5: 本文表の選定と図表整形

期間:

```text
2026-06-09 〜 2026-06-15
```

### 作業

- `synthetic_component_recovery` から本文用の 6〜9 行を選ぶ。
- `freshretail_correction` から `series_mean_all` と `same_hour_recent_mean_d7_all` の主比較を作る。
- `statistical_validation` から本文用の CI 行だけを抜き出す。
- 2-Exp-21 の heatmap / hour profile 図を本文候補として選ぶ。
- appendix に回す表を決める。

### 完了条件

- 本文 Table 1〜3 の行数と指標が決まっている。
- Figure 1〜3 の候補が決まっている。
- 「本文に載せる結果」と「appendix に回す結果」が分かれている。

## Revised Week 6: 追加検証 1 - 系列数感度

期間:

```text
2026-06-16 〜 2026-06-22
```

### 実験

- `2-Exp-24`: FreshRetailNet scale sensitivity

### 目的

現在の FreshRetailNet 主結果は、最大 6000 train 系列、最大 1500 eval 系列での結果である。

2-Exp-24 では、系列数を 2000 / 6000 / 12000 に変えて、`series_mean` residual の改善が保たれるかを確認する。

### 完了条件

- `series_mean_12k` でも corrected MAE が baseline MAE より低い。
- `series_mean_12k` でも high residual top10 が改善する。
- `series_mean_12k` でも hour component が残差の時間帯 profile と対応する。
- `same_hour_recent_mean_d7_12k` では改善が小さく、強い baseline 下の限界例として扱える。

### 実行コマンド

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-24_freshretailnet_scale_sensitivity_smoke.json
uv run decoupled-ts residual-sweep --config configs/2-Exp-24_freshretailnet_scale_sensitivity.json
```

状態:

```text
完了。series_mean residual は 2k / 6k / 12k すべてで baseline を改善した。
series_mean_12k では baseline MAE 0.0694 に対し、corrected MAE は 0.0487〜0.0494。
hour component residual profile corr も 0.9822〜0.9920 と高く、系列数を増やしても hour 成分の対応は崩れなかった。
same_hour_recent_mean_d7 は MAE では小さく改善したが、hour corr が負になり、解釈可能な分解としては弱い。
```

## Revised Week 6.5: 追加検証 2 - 系列ブロック頑健性

期間:

```text
2026-06-20 〜 2026-06-27
```

### 実験

- `2-Exp-25`: FreshRetailNet block robustness

### 目的

2-Exp-24 は、先頭から取る系列数を増やす実験だった。

2-Exp-25 では、系列の開始位置をずらし、先頭ブロック以外でも `series_mean` residual の改善と hour component の対応が保たれるかを確認する。

### 完了条件

- `series_mean_block1_6k` と `series_mean_block2_6k` でも corrected MAE が baseline MAE より低い。
- high residual top10 でも改善する。
- hour component residual profile corr が正で高い。
- `same_hour_recent_mean_d7` では、MAE 改善と解釈可能性が分離するという限界解釈が保たれる。

### 実行コマンド

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-25_freshretailnet_block_robustness_smoke.json
uv run decoupled-ts residual-sweep --config configs/2-Exp-25_freshretailnet_block_robustness.json
```

状態:

```text
完了。series_mean residual は block0 / block1 / block2 のすべてで baseline を改善した。
MAE 改善幅は 0.0173〜0.0190 程度で、先頭系列だけに依存する結果ではなかった。
hour component residual profile corr も mae_grid_reference で 0.9754〜0.9957 と高く、時間帯成分の対応は保たれた。
same_hour_recent_mean_d7 は各 block で MAE がわずかに改善するが、hour corr は -0.8870〜-0.8085 と負であり、解釈可能な分解としては弱い。
```

## Revised Week 6.75: 橋渡し検証 - 元論文から提案手法への接続

期間:

```text
2026-06-18 〜 2026-06-24
```

### 実験

- `2-Exp-26`: 元論文準拠の `global/local` 2 成分から `global/day/hour/(interaction)` への拡張
- `2-Exp-27`: direct target と `series_mean` residual target の最小比較

### 目的

2-Exp-26 では、いきなり residual output decomposition を提案するのではなく、元論文の `global/local` 分離を小売需要へ移植し、local を day/hour/interaction に細分化するだけで何が起きるかを見る。

2-Exp-27 では、同じ比較を residual target に移し、direct target と residual target の違いを最小構成で確認する。
これにより、「元論文からの自然な拡張」と「残差側へ移る必要性」を同時に整理する。

### 結果の読み方

- 2-Exp-26 では、`four_factor_global_day_hour` が direct target で小さく改善したが、interaction 追加は安定しなかった。
- 2-Exp-27 では、residual target は baseline を改善したが、4 成分 latent が `global/local` residual を常に上回るわけではなかった。
- したがって、proposal では「4 成分 latent に分ければ良い」とは書かない。
- 書くべきことは、「元論文の global/local 分離は出発点として有効だが、小売需要の補正では、baseline 後の residual を output-level に分解し、centering constraints で成分の読みを固定する必要がある」である。

### 完了条件

- 2-Exp-26/27 の結果が `doc/proposal/Proposal-zemi.md` と実験ドキュメントに反映されている。
- Related Work で、latent representation learning と output decomposition の違いを説明できる。
- Discussion で、`residual なら常に良い` ではなく `baseline と residual structure に依存する` と書ける。

状態:

```text
完了。2-Exp-26/27 は、主性能を更新するためではなく、proposal の論理の飛躍を減らす橋渡し実験として使う。
```

## Revised Week 6.9: 応用例整理

期間:

```text
2026-06-28 〜 2026-07-03
```

### 作業

- 高残差改善を、異常検知・外れケース補正として整理する。
- hour component を、時間帯別発注・陳列・人員配置の補助情報として整理する。
- series component を、基準値自体の見直し候補として整理する。
- same-hour baseline で改善が小さいことを、強い基準値の診断例として整理する。

### 完了条件

- Discussion に「応用先」を 1 節追加できる。
- ゼミ資料・論文ともに、残差表現を学ぶ意義が予測精度以外にも説明できる。

## Revised Week 7-8: 執筆

期間:

```text
2026-06-23 〜 2026-07-06
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
2026-07-07 〜 2026-07-13
```

### 判断基準

#### Case A

2-Exp-20 で baseline 改善の CI が明確に 0 未満。

主張:

```text
残差分解は解釈可能性と予測補正の両方に寄与する。
```

現時点の位置:

```text
series_mean_all では Case A に寄せられる。ただし FreshRetailNet 全体で常に改善する主張にはしない。
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

現時点では、synthetic は Case A 相当、FreshRetailNet は条件付き Case A / Case B として書くのが妥当である。
