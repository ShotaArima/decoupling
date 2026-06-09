# 2-Exp-10: Factor-Structured Subset Evaluation

## 目的

`2-Exp-9` では FreshRetailNet subset でも `b + r_hat` の平均補正性能は same-hour baseline を明確には超えなかった。
そこで `2-Exp-10` では、予測 MAE の改善だけでなく、残差表現が因子別に意味を持つかを検証する。

中心問いは以下。

```text
FreshRetailNet 内で residual が global/day/hour/interaction のどの因子に由来するかを推定し、
対応する latent がその情報を持ち、対応しない latent には漏れにくいか。
```

## 追加した subset 設計

`residual-sweep` に `sweep.subsets` を追加し、1 つの config から複数 subset を実行できるようにした。

| subset | sort key | 狙い |
|---|---|---|
| `day_structured` | `weekday_variance_explained` | 曜日単位の residual 構造を持つ系列 |
| `hour_structured` | `hour_variance_explained` | 時間帯単位の residual 構造を持つ系列 |
| `interaction_structured` | `interaction_nonadditive` | day x hour の非加法成分を持つ系列 |
| `clean_active` | `baseline_mae` | 欠品・ゼロ起因ノイズを抑えつつ baseline が外す系列 |

各系列について same-hour baseline から residual を作り、系列内の group variance explained や非加法スコアを計算して subset を選ぶ。

## 追加した評価指標

### Positive Probe

- `probe_z_global_subgroup_accuracy`
- `probe_z_day_weekday_accuracy`
- `probe_z_day_discount_mae`
- `probe_z_hour_hour_accuracy`
- `probe_z_interaction_weekday_accuracy`
- `probe_z_interaction_hour_accuracy`
- `probe_z_interaction_weekday_hour_accuracy`

### Leakage Probe

- `leakage_z_global_discount_mae`
- `leakage_z_day_subgroup_accuracy`
- `leakage_z_hour_subgroup_accuracy`

低い方が望ましい leakage と、高い方が望ましい target probe を分けて見る。

### Targeted Ablation

学習済み model の latent をゼロ化して decode し、residual MAE の悪化量を見る。

- `ablation_zero_z_global_residual_mae_delta`
- `ablation_zero_z_day_residual_mae_delta`
- `ablation_zero_z_hour_residual_mae_delta`
- `ablation_zero_z_interaction_residual_mae_delta`

該当 subset で対応 latent の delta が最も大きければ、再構成上その latent が使われている根拠になる。

## 実行コマンド

Smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-10_factor_subsets_smoke.json
```

FreshRetailNet:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-10_factor_subsets_freshretailnet.json
```

## 出力

`sweep.output_dir` 配下に以下を出力する。

- `all_results.csv`: subset x seed x model の全結果。
- `aggregate.csv`: subset x model ごとの mean/std。
- `summary.json`: JSON 版の全結果。
- `<subset>_seed_<seed>/`: 通常の residual experiment 出力。

## 成功条件

予測性能だけでなく、以下を主な論拠にする。

- `day_structured` で `z_day` probe と `z_day` ablation delta が強い。
- `hour_structured` で `z_hour` probe と `z_hour` ablation delta が強い。
- `interaction_structured` で `z_interaction` probe または ablation delta が相対的に強い。
- leakage probe が target probe より弱い。
- `interaction_with_swap` が同等の reconstruction を保ったまま leakage を減らす、または対応 probe/ablation を改善する。

FreshRetailNet で予測補正が小さい場合でも、この条件が満たされれば「予測器としての改善」ではなく「残差構造の因子分解表現」として論拠を積める。
