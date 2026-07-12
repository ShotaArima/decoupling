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

## 実行結果(2026-07-13、リモート)

### (1) 窓内 FreshRetailNet(Exp-32 同一設定、2000/500系列)

empirical ANOVA は決定論的なため5 seed で同一値(std 0)。NN は5 seed 平均±std。

| model | corrected MAE | top10 MAE | corrected WAPE | residual R2 | bias |
|---|---:|---:|---:|---:|---:|
| baseline のみ | 0.0721 | 0.2923 | 1.166 | — | ~0 |
| **empirical ANOVA(主効果)** | **0.0555** | **0.1817** | **0.898** | **0.444** | **~0** |
| NN(中心化・全成分) | 0.0583±0.0011 | 0.2508±0.0154 | 0.942 | 0.248 | −0.187±0.132 |

**経験的 ANOVA が窓内で NN を明確に上回った**(MAE で −4.7%、top10 で −27.5%)。
bias もほぼ 0(NN は −0.19 の系統的偏り)。

注意: empirical の `hour_component_residual_profile_corr = 1.0000` は
**トートロジー**である(hour 成分 = 残差の hour 平均 profile そのものなので、
自分自身との相関)。解釈性の証拠として使ってはならない。

### (2) 未来日 FreshRetailNet(Exp-30 同一設定、seed 13)

| model | future baseline MAE | future corrected MAE | 改善率 | future hour corr |
|---|---:|---:|---:|---:|
| empirical ANOVA | 0.0731 | **0.0589** | **19.5%** | 0.985 |
| NN(中心化, int なし)※Exp-30 | 0.0731 | 0.0594 | 18.8% | 0.987 |
| NN(中心化, 全成分)※Exp-30 | 0.0731 | 0.0601 | 17.8% | 0.961 |

未来日でも **empirical が NN と同等(僅差で上)**。FreshRetailNet の
`series_mean` 残差で未来日へ転移する構造は実質 hour 主効果+系列水準であり、
これは平均の持ち越しで十分に推定できる。

### (3) 未来日 synthetic(Exp-30 synthetic 同一設定、3 seed)

| model | future corrected MAE | 改善率 | future day corr | interaction corr(窓内) | true 残差への MAE(窓内) |
|---|---:|---:|---:|---:|---:|
| empirical ANOVA | 0.3544±0.0013 | 48.7% | **0.0** | **0.0** | 0.0725 |
| NN(中心化, 全成分)※Exp-30 | **0.1028±0.0012** | **85.1%** | **0.983** | 0.979 | **0.0293** |

synthetic では **NN が圧勝**。empirical は (i) 未来日の day 成分を原理的に
出せず(day corr = 0、day MAE 0.333)、(ii) interaction を持たず(corr 0)、
(iii) 窓内でも per-day 平均のノイズにより day corr 0.937 に留まる
(NN は系列間 pooling と特徴量で 0.999)。真のノイズなし残差への近さでも
NN が 2.5 倍正確(0.0293 vs 0.0725)。

### (4) 加法性診断(7(a))

| データ | `diag_scale_dependence_corr` | 読み |
|---|---:|---|
| synthetic(加法生成) | 0.03±0.07 | 加法構造を正しく検出(診断の妥当性確認) |
| **FreshRetailNet** | **0.971**(log でも 0.980) | **残差の大きさは売上水準にほぼ比例 = 乗法的構造が強く示唆される** |

## 読み取り(事前分岐の「窓内で匹敵/上回る」ケースが発動)

1. **FreshRetailNet の `series_mean` 残差は、主効果(hour+水準)でほぼ説明が尽きる**。
   その推定に NN は不要で、マスク付き平均の方が正確かつ無 bias。
   NN は表現のボトルネックを介するぶん、主効果の推定では平均に勝てない。
2. **NN の必然性は「特徴量で駆動される day / interaction 構造が存在する場合」に限定される**。
   synthetic がその条件で、未来日転移(85% vs 49%)・interaction 回復(0.98 vs 0)・
   day の pooling 推定(0.999 vs 0.937)のすべてで NN が必要だった。
3. したがって枠組みは**ラダー(段階適用)**として提示するのが誠実:
   基準値 → 残差軸診断 → **経験的主効果補正(既定の第一選択)** →
   特徴駆動の day/interaction 構造がある場合に NN 成分モデル。
   FreshRetailNet は診断が「hour 主効果のみ」を示す条件であり、第一選択で足りる。
   これは提案の否定ではなく適用条件の精密化である(C2 の latent vs 出力の
   比較は NN 設計内の主張なので影響を受けない)。
4. **加法性の前提は FreshRetailNet では成立が疑わしい**(scale corr 0.97)。
   3.2 の「第一近似」を limitation に接続し、log 残差版を今後の課題とする。
   synthetic 側の 0.03 は診断自体の妥当性確認になっている。

## 論文への反映(要議論 — 主張の再構成を伴う)

- Table 2 に empirical ANOVA 行を追加すると、FreshRetailNet 窓内では提案 NN が
  素朴分解に負ける表になる。誠実な取り込み方は上記ラダーの明示:
  「主効果で尽きる条件では素朴分解で十分(診断がそれを教える)。NN は
  特徴駆動の day/interaction がある条件で必要(synthetic で実証)」。
- 未来日表(tab:future_day)への行追加は NN 優位の文脈(synthetic)なので単純追加可。
- FreshRetailNet の各所の主張(「NN 補正が効く」)は「主効果補正が効く。
  その最良の推定器は条件に依る」へトーン調整が必要。
- scale 依存 0.97 の報告と limitation 追記。
- empirical の hour corr = 1 はトートロジーなので表に載せる場合は n/a 扱いか脚注。
