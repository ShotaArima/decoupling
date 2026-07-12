# 2-Exp-30: 未来日に対する成分出力の評価

## 背景

外部レビュー(2026-07-12)で、論文の最大の論理ギャップとして次が指摘された。

```text
「平均による分解は事後的であり、未来の日には直接適用できない。
そこで NN で各成分を予測する」と述べているが、
未来の日の成分をどう作るのか、定式化も実験も追いついていない。
日成分は予測不能なショックか、曜日・販促などの観測可能な特徴で
説明できる部分か、どちらを狙っているのか。
```

紙面上は (a)〜(d) の修正で「本研究の検証範囲は窓内推定と系列方向汎化」と
正直に限定した(paper リポジトリ review_notes 参照)。
2-Exp-30 は、この限定を一歩進めて「**事前に分かる特徴だけから未来日の成分を
出せるか**」を実験で直接検証する。

## 目的

観測窓の最後 $k=2$ 日を「未来日」とみなし、その日の売上情報を入力から隠した状態で、

1. 日成分・相互作用成分が、日単位特徴(曜日・休日・値引き・天気)と窓の文脈から構成できるか
2. 未来日セルに対する補正(baseline + r̂)が、文脈のみから計算した基準値を上回るか

を評価する。これは「日成分 = 観測可能な特徴で説明できる日効果」という
論文の位置づけの実験的裏づけになる(論証マップの R3 介入応答にも接続)。

## 設計

### 入力マスク

- config の `residual.future_mask_days = k` で窓の最後 $k$ 日を未来日に指定する。
- 未来日のセルについて、`residual.future_mask_channels`(既定 `[0, 1]` =
  売上/残差チャネルと欠品・在庫チャネル)の **x とマスクの両方をゼロ化**する。
  日単位特徴(値引き・休日・天気)と時刻・曜日特徴は残す。
- 中心化は従来どおり窓全体($D$ 日)で課す。モデルは全日分の成分を出力する。

### リーク防止(重要)

- 基準値は**文脈日(最初の $D-k$ 日)の観測セルのみ**から計算する
  (`series_mean` は文脈平均になる。`same_hour_recent_mean` の rolling window も
  未来日を除外することをユニットテストで確認済み)。
- 残差 $r = y - b$ はこの文脈基準値に対して全日で定義し、損失は真の観測マスク上で
  全日を監督する(未来日は特徴のみから当てる訓練になる)。
- synthetic(`noisy_true_residual` target)では残差は生成的に定義されるため
  基準値のリークは問題にならない。入力マスクのみが効く。

### 評価指標(新設 `future_holdout_metrics`)

| 指標 | 意味 |
|---|---|
| `future_baseline_cell_mae` / `future_corrected_cell_mae` | 未来日セルでの基準値のみ vs 補正後 |
| `context_*` | 同じ指標の文脈日版(参照) |
| `future_residual_hour_profile_corr` / `future_hour_component_residual_profile_corr` | 未来日セルでの時間帯 profile 対応 |
| `future_component_day_corr` / `future_component_interaction_corr` | (synthetic)未来日の推定成分と真の成分の相関 |

## 成功条件

強い結果:

- synthetic: `future_component_day_corr` が高い(真の day 効果は曜日・値引き・休日の
  関数なので、特徴から復元できるはず)。interaction も同様(値引き×13時 promo)。
- `future_corrected_cell_mae < future_baseline_cell_mae`(未来日でも補正が効く)。

この場合、論文の limitation(未来日外挿は未検証)を1段落+小表で
「特徴から構成できる範囲では外挿も成立する」に格上げできる。

弱い結果:

- 未来日の補正が baseline と同等以下 → 「窓内推定と系列方向汎化」の限定を維持し、
  本実験は appendix / 修論行き。day 成分のうち特徴で説明できない部分
  (予測不能ショック)が支配的である、という知見として整理する。

## 実装

- `src/decoupled_ts/residual_experiments.py`
  - `residual_batch`: `future_mask_days` / `future_mask_channels` 対応。
    基準値計算を `baseline_observed`(文脈のみ)に変更。入力 x/mask の未来日
    ゼロ化。`future_cells` を返す。損失・評価用の `observed` は真のマスクのまま。
  - `predict_residuals`: `future_cells` を配列として収集。
  - `future_holdout_metrics`: 上記指標を計算。`run_variant` から呼び出し。
- 既存実験への影響: `future_mask_days` 未指定(=0)なら従来と完全に同一経路。

## 実行コマンド

smoke(ローカル実行済み・完走確認済み):

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-30_future_day_component_smoke.json
```

synthetic 本番(3 seed。リモート `ssh my` で実行):

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-30_future_day_component_synthetic.json
```

FreshRetailNet(seed 13、Exp-28 と同一規模 2000/500 系列):

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-30_future_day_component_freshretailnet.json
```

## 出力

```text
runs/2-Exp-30_future_day_component_synthetic/base/seed_{17,23,31}/{variant}/metrics.json
runs/2-Exp-30_future_day_component_freshretailnet/{variant}/metrics.json
```

variant: `paper_global_local_residual`(参照) /
`output_decomp_centered_no_interaction` / `output_decomp_centered`

見るべきキー: `future_*` と `context_*`(metrics.json 内)。

## ユニット検証(ローカル、2026-07-12)

- future_cells の位置・形状が正しい
- 未来日の ch0/ch1 の x とマスクが両方ゼロ化され、特徴チャネルは無傷
- `series_mean` 基準値が文脈日のみの平均と一致(リークなし)
- `same_hour_recent_mean` の rolling window が未来日を除外
- 損失・評価用 `observed` は真の観測マスクのまま
- `future_mask_days` 未指定時は従来挙動と同一(future キーなし)
- smoke 完走: future/context 指標が metrics.json に出力される
  (1 epoch でも `future_component_day_corr = 0.43`)

## 論文での使い方

結果が強ければ: 5章に1段落+小表(future MAE 対比、future day corr)を追加し、
6章の限界第4項を「特徴から構成できる範囲では未来日にも成分を出力できる」に更新。
結果が弱ければ: 本文は現状の限定を維持し、修論・中間発表の材料にする。

## FreshRetailNet 結果(2026-07-13, seed 13, 2000/500系列)

実行コマンド:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-30_future_day_component_freshretailnet.json
```

### 未来日 vs 文脈日の補正性能

`baseline_cell_mae`(context 0.0721 / future 0.0731)は文脈日のみから計算した
`series_mean` である。future の baseline がわずかに高いのは，直近の売上情報が
使えないため水準推定の分散が増えるためと考えられる。

| model | future baseline MAE | future corrected MAE | 改善率 | context baseline MAE | context corrected MAE | 改善率 |
|---|---:|---:|---:|---:|---:|---:|
| `paper_global_local_residual` | 0.0731 | 0.0628 | 14.2% | 0.0721 | 0.0614 | 14.9% |
| `output_decomp_centered_no_interaction` | 0.0731 | **0.0594** | **18.8%** | 0.0721 | **0.0574** | 20.4% |
| `output_decomp_centered` | 0.0731 | 0.0601 | 17.8% | 0.0721 | 0.0583 | 19.1% |

3モデルすべてが，売上情報を一切与えられない未来日においても baseline を明確に
上回って補正している。改善率は文脈日よりわずかに小さいが（例:
`output_decomp_centered_no_interaction` で 20.4% → 18.8%），大きくは劣化していない。
モデル間の順序も主実験（2-Exp-28）と同じく `output_decomp_centered_no_interaction`
が最良であり，中心化を伴う成分別出力の優位性は未来日の設定でも保たれている。

### 時間帯成分の未来日への転移

| model | context hour profile corr | future hour profile corr | context hour component corr | future hour component corr |
|---|---:|---:|---:|---:|
| `paper_global_local_residual` | 0.8459 | 0.8729 | n/a | n/a |
| `output_decomp_centered_no_interaction` | 0.9986 | 0.9877 | 0.9958 | 0.9867 |
| `output_decomp_centered` | 0.9462 | 0.9605 | 0.9332 | 0.9606 |

`output_decomp_centered_no_interaction` では，時間帯成分と残差の時間帯 profile の
対応が未来日でも corr 0.987〜0.988 と高い水準で保たれている。
これは，時間帯成分が特定の観測セルへの過学習ではなく，曜日・季節性などの
繰り返し構造として学習されており，売上を見ていない未来日にも転移することを示す。

### 解釈と留保

この結果は，日成分・相互作用成分が「窓内の観測セルでしか意味を持たない」という
懸念に対して部分的な反証になる。すなわち，売上情報を伏せても baseline を上回る
補正が可能であり，特に時間帯成分は高い相関で未来日へ転移する。

ただし，本実行は FreshRetailNet（実データ）であるため，真の day 成分・
interaction 成分が存在せず，`future_component_day_corr` のような
ground truth との相関は計算できない（この指標は synthetic 実行でのみ得られる）。
したがって，「日成分が曜日・値引き・休日の効果を正しく捉えているか」という
定式化上の核心的な問いには，2-Exp-30 synthetic 実行の結果を待つ必要がある。
現時点では MAE と hour profile corr の観点でのみ，未来日への一般化を確認できた。

### 副次的な確認事項

本実行の `corrected_cell_mae`（全体，例: `output_decomp_centered_no_interaction`
で 0.05751）は，2-Exp-28（同一 seed 13，future masking なし）の対応値 0.05719
とわずかに異なる。これは baseline の計算対象日数が 28 日から文脈日 26 日に
変わったことによる差であり，モデル間の順序や結論には影響しない。

## Synthetic 結果(2026-07-13, seeds 17/23/31, 1500系列)

実行コマンド:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-30_future_day_component_synthetic.json
```

以下は 3 seed の aggregate(mean ± std)。

### 未来日の補正性能: latent 型は崩壊し，成分別出力は保たれる

| model | context corrected MAE | future corrected MAE | future baseline MAE | future 改善率 | context→future 劣化 |
|---|---:|---:|---:|---:|---:|
| `paper_global_local_residual` | **0.0173 ± 0.0009** | 0.598 ± 0.020 | 0.691 ± 0.022 | 13.5% | +3354% |
| `output_decomp_centered_no_interaction` | 0.1076 ± 0.0009 | 0.1212 ± 0.0023 | 0.691 | 82.5% | +12.6% |
| `output_decomp_centered` | 0.0970 ± 0.0007 | **0.1028 ± 0.0012** | 0.691 | **85.1%** | **+6.0%** |

**最重要の発見**: `paper_global_local_residual` は文脈日では MAE 0.0173 と
圧倒的に見えるが，未来日では 0.598 まで崩壊し，baseline 比の改善は 13.5% しかない。
一方 `output_decomp_centered` は文脈日 0.0970 → 未来日 0.1028 と**ほぼ劣化しない**
(+6.0%)。

### なぜか: noise floor による診断

このデータの学習 target(`noisy_true_residual`)には標準偏差 0.12 の観測ノイズが
含まれるため，構造だけを完全に当てた場合の理論下限(noise floor)は
mean|N(0, 0.12²)| ≈ **0.0957** である。

- 成分別出力モデルの文脈日 MAE(0.097〜0.108)は noise floor にほぼ張り付いており，
  「予測可能な構造をすべて捉え，ノイズは追わない」正しい挙動を示す。
- `global/local` の文脈日 MAE 0.0173 は **noise floor を大きく下回る**。
  これはノイズまで再現しているということであり，cell 単位の local latent が
  入力に含まれる観測残差(ノイズ込み)をほぼ恒等写像的に複製していることを意味する。
  観測を隠した未来日で baseline 水準まで崩壊するのはその帰結である。

真の(ノイズなし)残差に対する誤差 `component_total_true_residual_mae` でも
これは裏づけられる: 文脈込みの窓全体で `output_decomp_centered` は **0.0293**，
`global/local` は **0.1220**。つまり **noisy target 上の見かけの MAE では
global/local が勝つ(0.050 vs 0.097)が，真の信号への近さでは成分別出力が
4 倍以上正確**である。窓内評価だけでは「複製」と「構造の学習」を区別できない，
という方法論上の知見でもある。

### 成分は特徴量から未来日に構成できる(ground truth 検証)

| model | future day corr | future interaction corr | future residual hour profile corr |
|---|---:|---:|---:|
| `paper_global_local_residual` | n/a | n/a | 0.656 ± 0.457(不安定: 0.13/0.88/0.96) |
| `output_decomp_centered_no_interaction` | 0.957 ± 0.012 | n/a | 0.889 ± 0.032 |
| `output_decomp_centered` | **0.9825 ± 0.0012** | **0.9820 ± 0.0015** | **0.9976 ± 0.0015** |

売上・在庫を一切見ていない未来日について，`output_decomp_centered` の
日成分は真の日効果と corr 0.983，相互作用成分は真の相互作用と corr 0.982 で一致した。
真の日効果は曜日・値引き・休日・天気の関数として生成されているので，これは
**「日成分・相互作用成分は観測可能な日単位特徴から構成できる」ことの ground truth
による直接の確認**である(外部レビュー指摘への実験的回答)。

また，synthetic では真の相互作用が存在するため interaction head の有無が
未来日 MAE に効いている(0.1212 → 0.1028，約 15% 差)。interaction 成分の
価値が「特徴駆動の相互作用が存在し，かつ観測が使えない条件」で最も明確になる
という位置づけが得られた。

### 解釈のまとめと条件

1. 外部レビューの問い「日成分は予測不能ショックか，特徴で説明できる部分か」への回答:
   **本手法の日成分は特徴で説明できる部分を捉えるものであり，特徴駆動の日効果で
   あれば未来日にも corr 0.98 で構成できる**。予測不能ショックはノイズ項に残る。
2. ただし synthetic は day 効果が構成上 100% 特徴駆動である。実データ
   (FreshRetailNet)の future 改善が 18.8% に留まるのは，実データの日効果に
   特徴で説明できない部分が含まれるためと整合的に説明できる。
3. latent 分割型の「窓内での高精度」は少なくとも synthetic ではノイズの複製を含む。
   2-Exp-28(FreshRetailNet 窓内)で global/local が拮抗して見えたことの
   解釈にも再考の余地を与える(ただし実データで複製と断定はできない。
   言えるのは「窓内評価だけでは両者を区別できず，未来日評価が区別する」こと)。

## 未確定・次のアクション

- [x] 2-Exp-30 synthetic 実行(3 seed): 完了。day corr 0.983 / interaction corr 0.982
  で特徴量からの未来日成分の構成を ground truth で確認。
- [x] FreshRetailNet 実行(seed 13): 完了。未来日でも 18.8% 改善,
  hour 成分は corr 0.987 で転移。
- [ ] 論文への反映: 5章に「未来日評価」の1段落+小表(synthetic の
  latent 崩壊 vs 成分別出力の保持 + FreshRetailNet の future 改善)を追加し,
  6章限界第4項を「特徴から構成できる範囲では未来日にも成分を出力できる」へ更新する。
  noise floor の議論は本文に入れる余地がなければ appendix または口頭発表用。
