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
| 2-Exp-7 | [2-Exp-7_swap_regularization_design.md](2-Exp-7_swap_regularization_design.md) | 反実仮想 swap 正則化あり/なしの分離性比較 |
| 2-Exp-8 | [2-Exp-8_structured_residual_synthetic_design.md](2-Exp-8_structured_residual_synthetic_design.md) | 真の residual 構造を持つ synthetic で仮説成立条件を検証 |
| 2-Exp-9 | [2-Exp-9_multiseed_and_subset_design.md](2-Exp-9_multiseed_and_subset_design.md) | `interaction_no_swap` / `interaction_with_swap` の複数 seed と FreshRetailNet subset 評価 |
| 2-Exp-11 | [2-Exp-11_output_decomposition_design.md](2-Exp-11_output_decomposition_design.md) | 出力成分を `global/day/hour/interaction` に分ける残差直交分解モデル |
| 2-Exp-12〜15 | [2-Exp-12_to_15_followup_design.md](2-Exp-12_to_15_followup_design.md) | FreshRetailNet subset 条件、bias 制御、synthetic 難易度、論文用最終比較 |
| 2-Exp-16 | [2-Exp-16_residual_target_sensitivity.md](2-Exp-16_residual_target_sensitivity.md) | FreshRetailNet の residual target 感度、再現性 subset、集約粒度の比較 |
| 2-Exp-17 | [2-Exp-17_freshretailnet_hour_component_validation.md](2-Exp-17_freshretailnet_hour_component_validation.md) | FreshRetailNet での `series_mean` 残差と hour 成分寄与の 5 seed 検証 |
| 2-Exp-18 | [2-Exp-18_calibration_shrinkage.md](2-Exp-18_calibration_shrinkage.md) | FreshRetailNet 残差補正の validation calibration と shrinkage |
| 2-Exp-19 | [2-Exp-19_bias_constrained_calibration.md](2-Exp-19_bias_constrained_calibration.md) | FreshRetailNet 残差補正における bias 制約つき calibration |
| 2-Exp-20 | [2-Exp-20_statistical_validation.md](2-Exp-20_statistical_validation.md) | FreshRetailNet 残差補正の seed-level paired bootstrap |
| 2-Exp-21 | [2-Exp-21_visualization.md](2-Exp-21_visualization.md) | FreshRetailNet 残差補正の hour profile と成功・失敗例の可視化 |
| 2-Exp-22 | [2-Exp-22_synthetic_difficulty_final.md](2-Exp-22_synthetic_difficulty_final.md) | Synthetic での成分回復と失敗条件の最終表 |
| 2-Exp-23 | [2-Exp-23_paper_tables.md](2-Exp-23_paper_tables.md) | 論文用 final table の自動集約 |
| 2-Exp-24 | [2-Exp-24_freshretailnet_scale_sensitivity.md](2-Exp-24_freshretailnet_scale_sensitivity.md) | FreshRetailNet の系列数感度確認 |
| Proposal | [proposal/README.md](proposal/README.md) | 残差直交分解に基づく論文化方針、数理保証、実験計画 |
