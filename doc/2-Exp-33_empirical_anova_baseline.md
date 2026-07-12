# 2-Exp-33: 経験的 ANOVA ベースライン + 加法性診断

## 背景

外部レビュー(指摘5・7)より:

```text
5. 提案法の実体は「二元配置の主効果+交互作用分解を NN で予測可能にしたもの」。
   最も自然な比較対象は、観測残差にそのまま ANOVA 分解を適用し、
   時間帯成分と系列成分を未来へ持ち越す素朴なベースラインである。
   「NN である必要性」はこの素朴分解に勝ることで初めて正当化される。
7. 式(3)の加法分解の妥当性検証がない。乗法的なズレの可能性に触れるべき。
```

## 設計上の重要な発見: 完全な経験的 ANOVA は窓内で恒等写像に退化する

u まで含む完全な経験的 ANOVA(g+a+c+u)は、完全観測の窓では観測残差を
厳密に再構成する(r̂ = r、窓内 MAE = 0)。これは 2-Exp-30 で latent 型が
「ノイズごと複製」して未来日で崩壊したのと同型の現象であり、
**窓内評価だけでは素朴分解と構造学習を区別できない**という既存の主張を裏付ける。

したがって公平な素朴ベースラインは:

- **窓内**: 主効果のみ(g + a + c、u = 0)。観測セルのマスク付き平均から計算
- **未来日**: 観測がない日の day 効果は自然に 0 になる = 「系列成分と時間帯成分を
  未来へ持ち越す」というレビュアーの言う素朴分解そのもの

NN の必然性は未来日設定で示される見込み: NN は特徴量から未来日の a, u を
構成できる(2-Exp-30: day corr 0.98)が、経験的分解は原理的にできない
(day 効果 = 0、day corr = 0)。

## 実装

- `src/decoupled_ts/residual_models.py`: `EmpiricalAnovaResidualModel`
  - パラメータなし(dummy パラメータで学習ループ互換)。入力グリッドの
    残差チャネルからマスク付き平均で g / a_d / c_h を計算し r̂ = g+a+c を出力。
    interaction 成分は 0。観測ゼロの日・時間帯は効果 0(未来日マスクに自然対応)。
  - variant type: `empirical_anova`
- 単体検証済み: 完全観測で厳密な ANOVA 主効果と一致 / 未来日マスクで
  day 効果 0・hour と global は文脈のみから計算 / backward 互換(AMP対応)。

### 7(a) 加法性(スケール依存)診断

`residual_axis_diagnostics` に追加:

- `diag_scale_dependence_corr`: 系列ごとの平均絶対残差と平均売上の Pearson 相関
- `diag_scale_dependence_log_corr`: 両者の log1p 変換後の相関

単体検証: 乗法的残差(水準比例)で corr 0.99、加法的残差で 0.03 と正しく判別。
相関が小さければ「加法分解で十分」の実証、大きければ log 変換の動機として
正直に報告する(論文 3.2 に「加法は第一近似、乗法的な場合は log 変換で
同枠組み」を記載済み)。

## 実行コマンド

smoke(ローカル完走済み):

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-33_empirical_anova_smoke.json
```

リモート(`ssh my`)で3本:

```bash
# (1) 窓内: Exp-32 と同一設定(2000/500系列, train_holdout 検証分離, 5 seed)
#     empirical_anova + output_decomp_centered の paired 比較
uv run decoupled-ts residual-sweep --config configs/2-Exp-33_empirical_anova_freshretailnet.json

# (2) 未来日 FreshRetailNet: Exp-30 と同一設定(seed 13)。Exp-30 の NN 結果と比較
uv run decoupled-ts residual-experiment --config configs/2-Exp-33_empirical_anova_future_freshretailnet.json

# (3) 未来日 synthetic: Exp-30 synthetic と同一設定(3 seed)。day corr の対比用
uv run decoupled-ts residual-sweep --config configs/2-Exp-33_empirical_anova_future_synthetic.json
```

(1) は Exp-32 と seed・データ分割が同一なので、既存の Exp-32 の6モデルとも
paired 比較できる。empirical_anova は決定論的(seed 非依存)なので
5 seed の値は同一になる想定(パイプライン整合のため sweep のまま実行)。

## 見るべき出力

| 実行 | 指標 | 問い |
|---|---|---|
| (1) 窓内 | corrected MAE / top10 / hour corr | 窓内で素朴分解にどこまで迫られるか(強くても想定内) |
| (2)(3) 未来日 | future corrected MAE / future_component_day_corr | NN の必然性: 経験的分解は day corr = 0 のはず |
| 全部 | `diag_scale_dependence_corr` / `_log_corr` | 加法性の妥当性(小さければ加法で十分) |

## 結果の読み方(事前に決めておく分岐)

- 窓内で empirical が NN に匹敵/上回る場合: 「窓内の transductive な補正では
  素朴分解が強い。NN の価値は (i) 未来日への特徴量ベースの成分構成、
  (ii) 欠測・少数観測での安定性、(iii) 系列間 pooling にある」と精密化。
  窓内比較は素朴分解が評価セルの平均を直接使う点で有利(半 in-sample)である
  ことも明記する。
- 未来日で NN が明確に勝つ場合(想定): 「NN である必要性」への直接回答として
  表1行+本文数文を論文に追加。
- scale corr が大きい場合: 3.2 の「第一近似」を limitation に接続し、
  log 変換版を今後の課題(修論)へ。

## 論文への反映(結果確認後)

- Table 2(tab:latent_vs_output)に empirical ANOVA 行を追加(5 seed 表記)
- 未来日表(tab:future_day)に empirical ANOVA 行を追加(day corr = 0 の対比)
- 03 に古典手法(STL・固定効果・階層予測)の段落を追加済み(bib も追加済み)
- 5章 実験設定に scale 依存診断の1文+数値
