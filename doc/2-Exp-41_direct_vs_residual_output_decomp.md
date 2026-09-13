# 2-Exp-41: 直接 y 予測版 output decomposition vs 残差版（分水嶺 A）

作成日: 2026-08-22
対応: [proposal/direct_y_and_unified_encoder_decoder.md](proposal/direct_y_and_unified_encoder_decoder.md)
の検証ラダー 2-Exp-41（仮説 H-A1 / H-A3 / H-A5）、JIMA-04 の limitation
「成分別出力モデルの direct 版比較がない」の解消
関連: [2-Exp-40_jima_main_results.md](2-Exp-40_jima_main_results.md)、
[2-Exp-27_direct_vs_residual_bridge.md](2-Exp-27_direct_vs_residual_bridge.md)、
[proposal/formulation.md](proposal/formulation.md)

## 背景

現行の主提案は残差 $r = y - b$ を 4 成分に分解する。一方、元論文
（Tonekaboni et al. 2022）は元系列を直接再構成しており、direct 版は元論文への
回帰である。鍵になるのは次の恒等式である。

```text
series_mean baseline の b は、two-way ANOVA でいう grand mean の経験推定にすぎない。
中心化制約は a/c/u の担当範囲を固定するため、水準の受け皿は g 成分に一意に決まる。
⇒ 残差版 = 「g の受け持ち分を b に固定した direct 版」
```

2-Exp-26/27 で direct 予測が不安定だったのは **latent split**（水準が day/hour
latent に混ざる）であり、中心化つき**出力分解**の direct 版は未検証だった。
本実験がその初の検証となり、成立すれば「基準値の後処理補正」から
「基準値を内包する生成モデル」への格上げ（方向 A）の分水嶺を越える。

## 実装（本 PR に含むコード変更）

`residual_experiments.py` の `residual_batch` に `baseline_method: "zero"` を追加した。

- `baseline = 0`、`residual = y` となり、既存パイプラインがそのまま
  「y の直接 4 成分分解」として動く。
- 入力チャネル 0（残差）は生の売上のままになる。
- 補正後予測は `ŷ = 0 + r̂ = r̂`、すなわちモデル出力そのもの。
- 既存の全 baseline_method と後方互換（default 挙動は不変）。

direct 条件では `baseline_cell_mae` は「常に 0 を予測したときの MAE ≈ 平均売上」
になるため改善率は意味を持たない。**比較は corrected 系指標（y スケール）で行う**
（仮説 H-A5: direct と residual の corrected_cell_mae はどちらも ŷ vs y の
セル MAE であり、直接比較できる）。

## 目的と仮説

**H-A1（主仮説）**: 中心化制約下では水準の受け皿が g に一意に固定されるため、
latent split で起きた不安定化は出力分解では起きない。direct 版
`output_decomp_centered` は residual 版と同等の corrected MAE を出す。

**負の対照**: direct 版 `output_decomp_no_center` は、水準の割り当てが不定に
なるため residual 版の no_center より劣化が大きい（= 中心化の価値が direct で
さらに際立つ）ことが予想される。

**empirical 行**: direct 条件の empirical ANOVA は「y の窓内二元配置分解
（grand mean + 主効果）」であり、series_mean baseline + 残差主効果と
数学的にほぼ同じものになるはずである（恒等式の閉形式での確認になる）。

## 実験条件

2-Exp-40 と同一条件（結果を直接並べるため）。

- dataset: FreshRetailNet、6000/1500/1500、train_holdout
- model: hidden 160、dims 10/10/10/8
- train: epochs 上限 60、early_stopping_patience 8、lr 1e-3
- seeds: 17, 23, 31, 47, 59

| scenario | baseline_method | 意味 |
|---|---|---|
| `residual_series_mean` | `series_mean` | 現行主提案（2-Exp-40 の再現） |
| `direct` | `zero` | y の直接 4 成分分解 |

| variant | 役割 |
|---|---|
| `empirical_anova_main_effects` | 閉形式の対応物（恒等式の確認） |
| `output_decomp_no_center` | 負の対照（水準混合の検出） |
| `output_decomp_centered` | 主比較 |

合計 run 数: 2 scenario × 5 seed × 3 variant = 30。

## 評価指標

すべて y スケールで比較する。

| 指標 | 見る理由 |
|---|---|
| `corrected_cell_mae` / `corrected_cell_wape` | 主判定（direct vs residual、同一スケール） |
| `high_residual_top10_corrected_mae` | 外れケースでの degrade 確認 |
| `corrected_cell_bias` | direct で水準 bias が出ないか |
| `component_*_mean_abs`（中心化違反） | direct でも centering が数値的に成立するか |
| `hour_component_residual_profile_corr` | 成分の質が direct で保たれるか |
| best/stopped epoch | direct は水準学習の分だけ収束が遅い可能性 |

## 事前に定める成功条件・分岐

1. **主判定（H-A1）**: direct 版 centered の corrected MAE が residual 版 centered
   に対して **paired 差 +0.0025 以内**（2-Exp-37 の安定判定と同じ幅）なら
   「direct でも成立（非劣性）」とする。さらに負けが 5 seed 中 1 以下なら
   「同等」と記載する。
2. **負の対照**: direct の no_center が direct の centered より明確に悪い
   （paired で 4/5 以上劣後）ことを確認する。これが出れば
   「中心化が direct 化の成立条件」という主張が立つ。
3. 非劣性が出ない場合: 劣化幅と bias / 収束速度を報告し、
   direct 化には尤度 decoder（2-Exp-42 案、H-A2）が必要と整理して
   残差版を主軸に維持する（撤退線）。

## 実行方法

smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-41_direct_vs_residual_smoke.json
```

本番:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-41_direct_vs_residual_freshretailnet.json
```

## 出力

```text
runs/2-Exp-41_direct_vs_residual_freshretailnet/
  residual_series_mean/seed_{17,23,31,47,59}/{3 variants}/
  direct/...
  all_results.csv / aggregate.csv / summary.json
```

## smoke 確認

2026-08-22 にローカル CPU で smoke（synthetic、2 scenario × 3 variant、seed 17）を実行し、
`baseline_method: "zero"` の経路を含めて完走を確認した。

## 本番結果

2026-08-22 にローカル GPU（RTX 5060 Ti）で全 30 run を実行し、完走した。

### scenario × variant 集計（5 seed の平均 ± SD、すべて y スケール）

| scenario | model | corrected MAE | top10 MAE | bias | hour corr | max stop |
|---|---|---:|---:|---:|---:|---:|
| residual | empirical ANOVA | 0.05443 | 0.1793 | +0.0000 | (1.0)* | 9 |
| residual | no_center | 0.05422 ± 0.00214 | 0.2306 | -0.308 | 0.993 | 55 |
| residual | **centered（現行主提案）** | **0.04828 ± 0.00032** | 0.1942 | -0.263 | 0.994 | 59 |
| direct | empirical ANOVA | 0.05443 | 0.1651 | +0.0000 | (1.0)* | 9 |
| direct | no_center | 0.04882 ± 0.00315 | 0.2193 | -0.500 | 0.983 | 60 |
| direct | centered | 0.05327 ± 0.00235 | 0.2664 | **-0.728** | 0.972 | 43 |

*empirical の hour corr はトートロジー（2-Exp-33 の注意と同じ）。

### paired 比較（corrected MAE、5 seed）

| 比較 | 平均差 | 勝敗 |
|---|---:|---|
| direct centered − residual centered | **+0.00499 ± 0.00225** | direct 0勝5敗 |
| direct: no_center − centered | -0.00445 | **no_center が 5勝0敗（残差版と逆転）** |
| residual: no_center − centered | +0.00594 | centered が 5勝0敗（2-Exp-40 の再現） |

### 事前に定めた成功条件との照合

1. **主判定（H-A1）: 不成立。** direct 版 centered は residual 版 centered に対して
   paired 差 +0.00499 で、事前の非劣性マージン +0.0025 を超えて劣後（0勝5敗）。
2. **負の対照: 予想と逆。** direct では no_center の方が centered より良い
   （残差版では centered が 5勝0敗なのに対し、完全に逆転した）。
3. 事前分岐 3（撤退線）を適用する: **残差版を主軸に維持**し、direct 化の道は
   下記の失敗解析に基づいて再設計する。

### 失敗の解析: 混合ではなく「g のボトルネック」

H-A1 は「中心化が水準の混合を防ぐので direct でも成立する」と予想した。
結果は、中心化は確かに混合を防いだが、**防いだことがそのまま失敗の原因**になった。

- direct + centered では、中心化により売上水準の全量が g 成分（系列ごとの
  スカラー 1 個を小さな head が出力）に押し込まれる。この g head は
  観測平均ほど正確に水準を推定できず、bias が -0.73 まで悪化した。
  水準は y の支配的成分なので、この誤差が MAE を直接押し上げた。
- direct + no_center が良いのは、水準が day/hour/interaction 成分へ自由に
  漏れられる（実質的に grid 全体で水準を表現できる）ため。ただし成分は
  読めなくなり、bias も -0.50 と大きい。「当たるが読めない」（risks.md R5）
  の direct 版が再現された。
- つまり direct 化の失敗は 2-Exp-26/27 の「latent の混合」とは別物で、
  **「水準（grand mean）の推定は NN の表現ボトルネックを通すと平均計算に勝てない」**
  という 2-Exp-38/39 で確立した原理の再現である。

### 帰結: 残差版の強い正当化

この結果は残差版の設計を「暫定的な選択」から「原理的に正しい役割分担」へ格上げする。

```text
水準（grand mean）＝ 閉形式の観測平均（= baseline b）が最良の推定器。
構造（day/hour/interaction のズレ）＝ 特徴量駆動の NN が最良の推定器。
残差版 y = b + r̂ は、この 2 つを最適な側に割り当てた hybrid そのものである。
```

direct 版を成立させるには、g を自由学習させるのではなく
(i) g 成分に閉形式推定（観測平均）を埋め込む — これは残差版と数学的に同値に近づく、
(ii) 尤度ベース decoder で水準をスケールパラメータとして扱う（H-A2、後続実験案）
のいずれかが必要である。(i) の同値性は、残差版が direct 版の特殊ケースである
という本実験の恒等式の、実証面からの裏返しになっている。

### 論文への反映

- JIMA-04 の limitation「成分別出力モデルの direct 版比較がない」は本実験で解消。
  「direct 版は水準推定がボトルネックとなり残差版に劣後する（+0.005、5 seed 全敗）。
  したがって残差化は前処理の便宜ではなく、水準を閉形式に委ねる設計上の選択である」
  と書ける。
- direct で centered/no_center の優劣が逆転する事実は、「中心化制約は
  残差ターゲットと組み合わせて初めて機能する」という適用条件として記述する。
- 方向 A（direct 化）は尤度 decoder（H-A2）を前提とした将来課題へ位置づけ直す。
