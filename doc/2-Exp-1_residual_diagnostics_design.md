# 2-Exp-1: 基準成分除去と残差構造の診断

## 目的

添付メモの中心仮説に合わせて、売上そのものではなく

```text
r = y - b
```

を表現学習の対象にする前処理診断を追加します。

この実験では、モデル学習より前に以下を確認します。

1. 売上 `y` は単純な基準成分 `b` でどの程度説明できるか。
2. `r = y - b` に、曜日・時間帯・系列差などの構造が残っているか。
3. 後続の残差再構成モデルで使うべき baseline をどれにするか。

## 新仮説の実験構成

この仮説は既存の `EXP-001`〜`EXP-007` とは別系列として扱い、`2-Exp-*` で番号を振ります。

| 実験 | 対応 Step | 目的 | 主な成功条件 |
|---|---|---|---|
| `2-Exp-1` | Step 1〜3 | 基準成分 `b` を定義し、`r = y - b` に構造が残るか確認する | 成功条件 1, 2 |
| `2-Exp-2` | Step 4 | `r` を対象に `z_global`, `z_day`, `z_hour` を学習する | 成功条件 3 の前提 |
| `2-Exp-3` | Step 5 | 残差再構成と単一 local / 多粒度 local の比較を行う | 成功条件 3 |
| `2-Exp-4` | Step 5 | ablation と probe task で潜在表現の役割分担を確認する | 成功条件 4 |
| `2-Exp-5` | Step 5 | heatmap / latent 可視化 / 反実仮想 swap で分離性を確認する | 成功条件 3, 4 |
| `2-Exp-6` | Step 6 | `b + r_hat` を下流の予測補正として評価する | 成功条件 5 |

成功条件は以下で固定します。

1. 基準成分 `b` が売上 `y` の大部分を説明する。
2. 残差 `r` に曜日・時間帯・店舗商品・欠品などの構造が残る。
3. 提案モデルが単一 local より `r` の再構成または解釈性で優れる。
4. `z_global`, `z_day`, `z_hour` が probe task で異なる情報を持つ。
5. 下流評価として `b + r_hat` が `b` のみより改善する、または少なくとも外れケースで改善する。

したがって `2-Exp-1` 単体で全成功条件を満たす必要はありません。`2-Exp-1` は成功条件 1 と 2 を確認し、以後の residual representation learning に進む根拠を作る実験です。

## 追加したコマンド

```bash
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-1_residual_diagnostics_smoke.json
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-1_residual_diagnostics_synthetic.json
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-1_residual_diagnostics_freshretailnet.json
```

## 追加したファイル

| file | 内容 |
|---|---|
| `src/decoupled_ts/residual_diagnostics.py` | 基準成分比較と残差構造分析の runner |
| `configs/2-Exp-1_residual_diagnostics_smoke.json` | synthetic small の動作確認 |
| `configs/2-Exp-1_residual_diagnostics_synthetic.json` | synthetic 本診断 |
| `configs/2-Exp-1_residual_diagnostics_freshretailnet.json` | FreshRetailNet 診断 |

## 比較する基準成分

| baseline | 意味 |
|---|---|
| `overall_mean` | 全観測セルの平均 |
| `series_mean` | 系列ごとの平均 |
| `recent_mean` | 直近 `recent_days` の平均 |
| `same_hour_recent_mean` | 直近 `recent_days` の同時間帯平均 |
| `weekday_same_hour_mean` | 過去の同曜日・同時間帯平均。履歴がない場合は同時間帯平均に fallback |

## 出力

root 出力先は `analysis.output_dir` です。

| file | 内容 |
|---|---|
| `baseline_metrics.json` | 各 baseline の MAE/RMSE/WAPE/R2/残差分散率 |
| `summary.json` | baseline 比較と選択 baseline の残差診断 |
| `residual_by_hour.csv` | 時間帯別の残差平均・絶対残差平均 |
| `residual_by_weekday.csv` | 曜日別の残差平均・絶対残差平均 |
| `residual_by_subgroup.csv` | subgroup 別の残差平均・絶対残差平均 |
| `residual_weekday_hour_heatmap.csv` | 曜日 x 時間帯の平均残差 heatmap |
| `run.log` | 実行ログ |

## Smoke 実行結果

実行コマンド:

```bash
uv run decoupled-ts residual-diagnostics --config configs/2-Exp-1_residual_diagnostics_smoke.json
```

主な結果:

| baseline | WAPE | R2 | residual variance ratio |
|---|---:|---:|---:|
| `overall_mean` | 0.6745 | 0.0000 | 1.0000 |
| `series_mean` | 0.5795 | 0.2374 | 0.7626 |
| `recent_mean` | 0.5829 | 0.2190 | 0.7805 |
| `same_hour_recent_mean` | 0.5359 | 0.3466 | 0.6529 |
| `weekday_same_hour_mean` | 0.6048 | 0.1436 | 0.8561 |

この smoke では `same_hour_recent_mean` が最も強く、以後の `r = y - b` 診断の初期 baseline として使います。

選択 baseline の残差構造:

| metric | value |
|---|---:|
| residual abs mean | 1.4360 |
| residual std | 1.9330 |
| observed cells | 28244 |
| linear probe R2 | 0.0576 |

`linear probe R2` は `discount`, `holiday`, `weather`, `hour`, `weekday`, `subgroup` から残差を説明する単純な線形回帰です。
本番データではこの値と `residual_by_*.csv` を見て、残差が完全なノイズかどうかを判断します。
