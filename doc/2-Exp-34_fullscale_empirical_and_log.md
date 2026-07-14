# 2-Exp-34: 経験的分解の 6000/1500 追試 + log 残差版(R2-1b / R2-5)

## 背景

外部レビュー第2ラウンド(2026-07-13)より:

```text
R2-1b: 表3(tab:freshretail_correction)と同じ 6000/1500 系列のスケールで
  経験的分解を1行追加すれば主張が完結する。閉形式なので計算コストはほぼゼロ。
R2-5: scale corr 0.97 は加法前提が実データでほぼ崩れていることを意味する。
  「今後の課題」送りは不可。series_mean の1条件だけでも log 版
  (r = log y − log b)を回し、加法版と結論が変わらないことを示すか、
  変わるならそれ自体を報告する。
```

## 設計

1つの sweep にまとめる: **6000/1500 系列 × 3 scenario × 2 variant × 3 seed**。

| scenario | baseline_method | 目的 |
|---|---|---|
| `series_mean_all` | `series_mean`(加法) | R2-1b: Table 3 スケールでの empirical 行。NN との対比 |
| `same_hour_recent_mean_d7_all` | `same_hour_recent_mean`(加法) | R2-1b: Table 3 のもう1条件 |
| `log_series_mean_all` | `log1p_series_mean`(新規実装) | R2-5: log 残差で結論が変わるか |

| variant | 内容 |
|---|---|
| `empirical_anova_main_effects` | 閉形式の主効果分解(枠組みの退化ケース。決定論的) |
| `output_decomp_centered` | NN 成分モデル(中心化・全成分) |

追加の設計判断:

- **`validation_source: train_holdout`** を採用(train 6000系列と重複しない
  train split の1500系列でモデル選択、公式 eval 1500系列で最終評価)。
  これにより旧課題 (f)(test = valid)がこのスケールでも解消される。
  ただし **Table 3 の既存値(Exp-19、test=valid・calibrationあり)とは選択手順が
  異なる**ため、行を追加する際は「同一スケール・選択分離済みの再実行」と明記する。
- diagnostics(permutation 500回 + scale 依存)を有効化。log 残差の
  A_hour・scale corr が加法版からどう変わるかも同時に取れる。
- log 版の評価は **level(元の売上)空間**で行われる:
  corrected = expm1(baseline_log + r̂)。baseline 自体は log 平均の逆変換
  (幾何平均に近い)なので、加法版の baseline MAE とは値が異なる。
  比較は「corrected 側の指標と結論(順序・hour corr・診断)」で行う。

## 実装

- `residual_batch` に `log1p_series_mean` を追加(log1p 空間の系列平均、
  `baseline_log` を返す既存の log1p_same_hour パターンを踏襲)。
- `future_holdout_metrics` を `baseline_log` 対応に修正(log 条件で未来日評価を
  行う場合も level 空間で正しく corrected を計算する。本実験では未使用だが整合のため)。
- 単体検証: baseline_log = 系列の log1p 平均、expm1(baseline_log + residual) が
  元売上を厳密に復元、future masking との併用も既存テストでカバー。
- smoke(synthetic、加法/log の2 scenario)ローカル完走。log 経路の
  level-space corrected MAE が加法版と同水準で出力されることを確認。

## 実行コマンド

smoke(ローカル実行済み):

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-34_fullscale_empirical_and_log_smoke.json
```

本番(リモート `ssh my`。18 run、GPU で30〜40分想定):

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-34_fullscale_empirical_and_log_freshretailnet.json
```

## 見るべき出力

`runs/2-Exp-34_fullscale_empirical_and_log/{scenario}/seed_{17,23,31}/{variant}/metrics.json`

| 問い | 指標 |
|---|---|
| R2-1b: 6000/1500 でも empirical が NN と同等以上か | `corrected_cell_mae` / `high_residual_top10_corrected_mae`(scenario 1, 2) |
| R2-5: log で結論が変わるか | 加法 vs log の `corrected_cell_mae`(level空間)、`hour_component_residual_profile_corr`、empirical vs NN の順序 |
| log で乗法性が解消されるか | `diag_scale_dependence_corr`(加法 0.97 → log で 0 付近に落ちるか) |
| log で hour 構造の診断が変わるか | `diag_hour_relative_amplitude` / `diag_hour_permutation_pvalue` |

## 結果の読み方(事前分岐)

- **log で scale corr が 0 付近に落ち、結論(empirical vs NN の順序、hour 構造の
  診断、補正の成立)が加法版と同じ** → 「加法版の結論は log 版でも保たれる。
  本文は加法で提示し、log の頑健性確認を1段落+表1行で報告」(最善)。
- **log で結論が変わる**(例: NN が empirical を上回る、hour 振幅が大きく変わる)
  → それ自体を正直に報告し、主結果をどちらのスケールで提示するか再議論。
  乗法構造下では加法の hour profile が高売上系列に引きずられるという
  レビュアーの懸念(R2-5 (i))が実証されたことになる。
- R2-1b は Table 3 に「経験的主効果分解(選択分離済み再実行)」行を追加し、
  ラダーの主張(FRN は第2段で足りる)を 6000/1500 スケールでも完結させる。

## 実行結果(2026-07-13、リモート。6000/1500系列・train_holdout・3 seed)

empirical は決定論的(3 seed で同一値)。NN は 3 seed 平均±std。
bias は相対値(Σ誤差/Σ|y|)。

### (1) series_mean(加法)— スケールで構図が変わった

| model | corrected MAE | 改善率 | top10 MAE | 相対 bias |
|---|---:|---:|---:|---:|
| baseline のみ | 0.0697 | — | 0.2788 | ~0 |
| empirical ANOVA | 0.0544 | 21.9% | **0.1793** | **~0** |
| NN(中心化・全成分) | **0.0516±0.0013** | **25.9%** | 0.2082±0.0112 | −0.217±0.054 |

**最重要の発見: 2000/500系列(Exp-33)では empirical が全指標で勝ったが、
6000/1500系列では NN が平均 MAE で逆転した(3 seed すべてで empirical 0.0544 を下回る)。**
一方、top10 は依然 empirical が大差で優位(0.1793 vs 0.2082。しかも Table 3 の
calibrated NN の最良値 0.1914 よりも良い)、bias も empirical はゼロ(構成上)。

→ 「窓内では empirical が上」という Exp-33 の結論は**学習系列数に依存**する。
正確な整理は: **平均誤差は学習データが増えるほど NN(系列間 pooling)が優位になり、
外れケース(top10)と無偏性は empirical が優位** — 両者は相補的。

### (2) same_hour_d7(加法)— 診断が「第2段すら止めるべき」と教えるケース

| model | corrected MAE | 改善率 | top10 MAE |
|---|---:|---:|---:|
| baseline のみ | 0.0580 | — | 0.2534 |
| empirical ANOVA | 0.0599 | **−3.3%(悪化)** | 0.2287 |
| NN | **0.0566±0.0006** | +2.4% | 0.2460 |

hour 構造が微小(A_hour=0.041、E≈3.3)な条件では、**経験的主効果補正は
baseline より悪化する**(微小構造の平均推定がノイズを足すだけ)。
NN は正則化された補正で小幅改善するが、bias は −0.019 → −0.173 に悪化。
これはラダーと診断の物語をさらに強める: **診断が弱い構造しか示さない場合、
経験的補正の適用自体を控えるべき**であり、その判定を A_k が与える。

### (3) log 残差版(R2-5)— 結論は不変、しかも両モデルとも改善

評価はすべて level(元の売上)空間。log 版の baseline は log 平均の逆変換
(幾何平均に近い。MAE 0.0682、bias −0.086)。

| model | corrected MAE(log版) | 加法版比 | top10 | hour corr |
|---|---:|---:|---:|---:|
| empirical ANOVA | 0.0510 | **−6.4%** | 0.1804(同等) | 1.0(トートロジー) |
| NN | **0.0484±0.0010** | **−6.2%** | 0.2024 | 0.995 |

- **結論の順序は完全に保存**: NN < empirical(平均 MAE)、empirical < NN(top10)、
  hour 軸が支配的(A_hour 0.527 vs 加法 0.542)、hour corr ~0.99、
  day 軸は検出されるが補正寄与小。
- **log 版は両モデルとも MAE を約6%改善** — 乗法性の指摘は方向として正しかった。
- ただし **scale corr は 0.953 → 0.802 までしか下がらない**。原因は売上値の
  スケールが小さく(mean|y|≈0.06)log1p がほぼ線形に振る舞うため。
  log1p 変換は乗法性を部分的にしか解消しない。

### 考察のまとめ

1. **ラダーの精密化(相補性)**: 「empirical vs NN のどちらが上」はスケールと
   指標に依存する。学習系列が十分なら平均誤差は NN、外れケースと無偏性は
   empirical。運用では両方を出して使い分ける(または NN + bias calibration)のが
   実際的で、これは枠組みがラダーであることの追加の論拠になる。
2. **診断の新しい実証**: same_hour 条件で経験的補正が悪化したことは、
   「A_k が小さい軸には補正を当てない」という判定規則の**負の対照を
   補正器側でも確認した**ことになる(これまでは解釈のみだった)。
3. **R2-5 は最善分岐**: 結論不変+性能改善。論文は加法で提示しつつ
   「log でも結論が保たれ、むしろわずかに改善する」ことを1段落で報告できる。
   限界の記述は「乗法性は log1p では部分的にしか解消されない(scale corr
   0.95→0.80)。完全な扱いは今後の課題」へ正確化する。

### 数値の取り扱い注意

- 本実行は train_holdout(選択と評価の分離)なので、Table 3 の既存値
  (test=valid、calibration あり)とは選択手順が異なる。empirical 行を Table 3 に
  足す場合は「選択分離済み・calibration なしの再実行」と注記する。
- NN(0.0516)と Table 3 の補正後 0.0501 の差は run 設定差(選択分離・seed)による。
- Exp-33(2000/500)の「窓内 empirical 優位」の記述は、論文では
  スケール依存であることを明示して更新する必要がある(下記)。

## 論文への反映(要更新箇所)

1. `subsec:empirical` の書き換え: 「窓内で empirical が NN を上回る」は
   2000/500 の結果。6000/1500 では平均 MAE が逆転し top10 は empirical 優位、
   という**相補性**として提示し直す(表に 6000/1500 列を追加 or 差し替え)。
2. Table 3 に empirical 行(0.0544 / top10 0.1793、注記付き)。
3. same_hour × empirical の悪化(0.0580→0.0599)を診断の負の対照として
   基準値感度 or 診断の節に1〜2文追加。
4. log 版の1段落(結論不変+6%改善+scale corr 0.80)と、
   06 限界第5項の更新(「log1p では部分的解消」)。
5. 07 課題第4項(log 版の実行)は完了済み内容になるため
   「乗法性の完全な扱い(スケール正規化等)」へ書き換え。
