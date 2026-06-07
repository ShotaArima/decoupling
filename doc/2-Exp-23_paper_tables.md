# 2-Exp-23: Paper Table Aggregation

## 背景

2-Exp-19 から 2-Exp-22 で、論文に必要な結果はほぼ揃った。

- FreshRetailNet では `series_mean_all` で予測補正が効く。
- 2-Exp-20 で seed-level の統計確認を行った。
- 2-Exp-21 で hour component の可視化を確認した。
- 2-Exp-22 で synthetic の成分回復と失敗条件を確認した。

2-Exp-23 は追加学習ではなく、これらの結果を論文用の表として固定する集約実験である。

## 目的

目的は 3 つ。

1. synthetic の成分回復表を固定する。
2. FreshRetailNet の補正性能と calibration trade-off を固定する。
3. 統計検証の CI 表を固定する。

これにより、論文本文と appendix の表を同じコマンドで再生成できるようにする。

## 入力

| 入力 | 内容 |
| --- | --- |
| `runs/2-Exp-22_synthetic_difficulty_final/summary.json` | synthetic difficulty の最終結果 |
| `runs/2-Exp-19_freshretailnet_bias_constrained_calibration/summary.json` | FreshRetailNet 補正と calibration |
| `runs/2-Exp-20_statistical_validation/summary.json` | seed-level paired bootstrap |

## 出力

| 出力 | 内容 |
| --- | --- |
| `synthetic_component_recovery.csv` / `.md` | synthetic の成分回復表 |
| `freshretail_correction.csv` / `.md` | FreshRetailNet の補正・bias・hour component 表 |
| `statistical_validation.csv` / `.md` | bootstrap CI と paired comparison 表 |
| `summary.json` | 生成された表の manifest |

## 成功条件

良い結果:

- 3 つの表が同じ出力ディレクトリに生成される。
- 表が `scenario` と `name` で比較できる。
- 本文に必要な主要指標が欠けていない。

悪い結果:

- 入力 summary の path が環境ごとにずれて再生成できない。
- 表に本文で必要な指標が不足している。
- synthetic と FreshRetailNet の役割分担が表から読み取れない。

## 実行コマンド

smoke:

```bash
uv run decoupled-ts paper-tables --config configs/2-Exp-23_paper_tables_smoke.json
```

本番:

```bash
uv run decoupled-ts paper-tables --config configs/2-Exp-23_paper_tables.json
```

## 論文での使い方

本文では次の対応にする。

| 論文上の表 | 2-Exp-23 の出力 |
| --- | --- |
| Synthetic main table | `synthetic_component_recovery.md` |
| FreshRetailNet correction table | `freshretail_correction.md` |
| Statistical validation table | `statistical_validation.md` |

2-Exp-23 が完了すれば、追加実験よりも執筆と図表整形に比重を移せる。
