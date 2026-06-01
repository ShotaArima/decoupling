# Experiment Documents

| Experiment ID | File | Summary |
|---|---|---|
| EXP-001 | [EXP-001_glr_freshretailnet_summary.md](EXP-001_glr_freshretailnet_summary.md) | 元論文寄りの local/global GLR 実験 |
| EXP-002 | [EXP-002_retail_multigrain_summary.md](EXP-002_retail_multigrain_summary.md) | 小売向け `z_global / z_day / z_hour / z_interaction` 分離実験 |
| EXP-003 | [EXP-003_hypothesis_validation_design.md](EXP-003_hypothesis_validation_design.md) | 多粒度分離仮説を検証するための実験設計 |
| EXP-004 | [EXP-004_to_EXP-007_next_experiment_design.md](EXP-004_to_EXP-007_next_experiment_design.md) | valid WAPE checkpoint と early stopping |
| EXP-005 | [EXP-004_to_EXP-007_next_experiment_design.md](EXP-004_to_EXP-007_next_experiment_design.md) | 30 epoch の synthetic / FreshRetailNet 本実験 |
| EXP-006 | [EXP-004_to_EXP-007_next_experiment_design.md](EXP-004_to_EXP-007_next_experiment_design.md) | naive baseline と proposed の比較可能性確認 |
| EXP-007 | [EXP-004_to_EXP-007_next_experiment_design.md](EXP-004_to_EXP-007_next_experiment_design.md) | 欠品重み、probe、hour heatmap |
| EXP-008 | [EXP-008_to_EXP-012_followup_experiment_plan.md](EXP-008_to_EXP-012_followup_experiment_plan.md) | FreshRetailNet の `same_hour_recent_mean` が強い理由の分析 |
| EXP-009 | [EXP-008_to_EXP-012_followup_experiment_plan.md](EXP-008_to_EXP-012_followup_experiment_plan.md) | 店舗×カテゴリ単位への集約 |
| EXP-010 | [EXP-008_to_EXP-012_followup_experiment_plan.md](EXP-008_to_EXP-012_followup_experiment_plan.md) | same-hour recent mean を residual baseline として組み込む |
| EXP-011 | [EXP-008_to_EXP-012_followup_experiment_plan.md](EXP-008_to_EXP-012_followup_experiment_plan.md) | naive baseline からの残差を予測対象にする |
| EXP-012 | [EXP-008_to_EXP-012_followup_experiment_plan.md](EXP-008_to_EXP-012_followup_experiment_plan.md) | weekday / holiday / discount / hour pattern 中心の probe 再設計 |
| 2-Exp-1 | [2-Exp-1_residual_diagnostics_design.md](2-Exp-1_residual_diagnostics_design.md) | 基準成分 `b` と残差 `r = y - b` の構造診断 |
| 2-Exp-2〜6 | [2-Exp-2_to_6_residual_representation_design.md](2-Exp-2_to_6_residual_representation_design.md) | 残差表現学習、再構成、probe、heatmap、`b + r_hat` 補正評価 |

