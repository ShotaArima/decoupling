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

## 論文への反映(結果確認後)

- Table 3 に empirical 行(scenario 1, 2)
- R2-5 の結果を 5章(1段落)+ 6章限界第5項の更新(「未確認」から「確認済み」へ)
- scale corr(log 版)の数値を診断の節に併記
