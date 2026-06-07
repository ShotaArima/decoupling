# 2-Exp-20: Statistical Validation of FreshRetailNet Residual Correction

## 背景

2-Exp-19 では、bias 制約つき calibration が validation 上の bias 制約を満たした。

結果の読み方は単純ではない。

- `mae_grid_reference` は全体 MAE が最も良い。
- `bias_constrained_001` は bias を大きく下げ、`series_mean_all` の high residual top10 で最も良い。
- `affine_reference` は bias を消せるが、全体 MAE が悪化する。

論文に載せるには、平均値だけでなく seed 間で安定しているかを見る必要がある。

## 目的

2-Exp-20 は、2-Exp-19 の 5 seed 結果に対して seed-level paired bootstrap を行う。

目的は次の 3 つ。

1. baseline に対する改善が seed 平均だけでなく安定しているか。
2. `bias_constrained_001` が `mae_grid_reference` より bias / high residual で良いと言えるか。
3. `bias_constrained_001` が `affine_reference` より全体 MAE を保てるか。

## 実装

追加 CLI:

```bash
uv run decoupled-ts residual-result-analysis --config configs/2-Exp-20_statistical_validation.json
```

入力は `runs/2-Exp-19_freshretailnet_bias_constrained_calibration/summary.json`。

出力:

| file | 内容 |
| --- | --- |
| `baseline_delta.csv` | 各 model が baseline よりどれだけ良いか |
| `paired_model_delta.csv` | model 間の seed 対応差 |
| `summary.json` | 上記の JSON 版 |

## 良い結果

- `series_mean_all / bias_constrained_001 / calibrated_corrected_cell_mae` の delta CI が 0 未満。
- `series_mean_all / bias_constrained_001 / calibrated_high_residual_top10_corrected_mae` の delta CI が 0 未満。
- `bias_constrained_001 - affine_reference` の `calibrated_corrected_cell_mae` が 0 未満。
- `bias_constrained_001 - mae_grid_reference` の `calibrated_high_residual_top10_corrected_mae` が 0 未満。

## 悪い結果

- baseline に対する CI が 0 をまたぐ。
- `bias_constrained_001` が high residual でも `mae_grid_reference` に勝たない。
- `affine_reference` と比べて MAE 優位がない。

## 実行コマンド

smoke:

```bash
uv run decoupled-ts residual-result-analysis --config configs/2-Exp-20_statistical_validation_smoke.json
```

本実験:

```bash
uv run decoupled-ts residual-result-analysis --config configs/2-Exp-20_statistical_validation.json
```

## 論文上の判断

2-Exp-20 で baseline 改善の信頼区間が 0 未満なら、FreshRetailNet でも「条件付きで予測補正として有効」と書ける。

もし CI が弱い場合でも、次の主張に落とせる。

```text
FreshRetailNet では series_mean residual に hour 構造が残り、提案分解はその構造を安定して抽出する。
ただし calibration の目的関数により、全体 MAE、bias、高残差改善の間には trade-off がある。
```

## 実行結果

`series_mean_all` の主要結果は次の通り。

| model | metric | baseline delta mean | 95% CI | improved runs |
| --- | --- | ---: | ---: | ---: |
| `mae_grid_reference` | `calibrated_corrected_cell_mae` | -0.0188 | [-0.0194, -0.0183] | 5/5 |
| `mae_grid_reference` | `calibrated_high_residual_top10_corrected_mae` | -0.0661 | [-0.0720, -0.0621] | 5/5 |
| `affine_reference` | `calibrated_corrected_cell_mae` | -0.0143 | [-0.0153, -0.0134] | 5/5 |
| `affine_reference` | `calibrated_high_residual_top10_corrected_mae` | -0.0745 | [-0.0790, -0.0688] | 5/5 |
| `bias_constrained_001` | `calibrated_corrected_cell_mae` | -0.0163 | [-0.0167, -0.0157] | 5/5 |
| `bias_constrained_001` | `calibrated_high_residual_top10_corrected_mae` | -0.0874 | [-0.0905, -0.0842] | 5/5 |

`bias_constrained_001` と代表 baseline の seed 対応比較は次の通り。

| comparison | metric | mean left-right | 95% CI | left better |
| --- | --- | ---: | ---: | ---: |
| `bias_constrained_001 - mae_grid_reference` | `calibrated_corrected_cell_mae` | 0.0025 | [0.0017, 0.0033] | 0/5 |
| `bias_constrained_001 - mae_grid_reference` | `calibrated_corrected_cell_bias` | -0.2695 | [-0.2971, -0.2419] | 5/5 |
| `bias_constrained_001 - mae_grid_reference` | `calibrated_high_residual_top10_corrected_mae` | -0.0213 | [-0.0267, -0.0163] | 5/5 |
| `bias_constrained_001 - affine_reference` | `calibrated_corrected_cell_mae` | -0.0019 | [-0.0032, -0.0006] | 4/5 |
| `bias_constrained_001 - affine_reference` | `calibrated_corrected_cell_bias` | 0.0048 | [0.0032, 0.0066] | 0/5 |
| `bias_constrained_001 - affine_reference` | `calibrated_high_residual_top10_corrected_mae` | -0.0129 | [-0.0197, -0.0060] | 5/5 |

## 考察

`series_mean_all` では、主要モデルすべてで baseline に対する改善 CI が 0 未満だった。したがって、FreshRetailNet でも条件を選べば予測補正として有効、という主張は可能である。

ただし、どの calibration を主モデルにするかは目的次第で変わる。

- 全体 MAE 最優先なら `mae_grid_reference`。
- bias と high residual top10 を重視するなら `bias_constrained_001`。
- bias 最小化だけなら `affine_reference` だが、全体 MAE は悪化する。

論文本文では `bias_constrained_001` を「外れケースと bias を重視した補正器」として扱い、`mae_grid_reference` を全体 MAE の上限性能として併記するのが妥当である。

## 次の実験

2-Exp-20 で数値主張は固まり始めた。

次は `2-Exp-21` として、hour profile と成功・失敗例の可視化を行う。これは新しい性能主張を増やすためではなく、論文図で「何を分解しているのか」を説明するための実験である。
