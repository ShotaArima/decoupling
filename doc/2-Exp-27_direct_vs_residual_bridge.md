# 2-Exp-27: Direct Target vs Residual Target Bridge

## 背景

2-Exp-26 では、元論文に近い `global + local` 分解を通常の future total sales forecasting に移植し、local を `day/hour/interaction` に分けた。

FreshRetailNet では、`global + day + hour` は `global + local` より小幅に改善した。

```text
paper_global_local WAPE = 0.6233
four_factor_global_day_hour WAPE = 0.6133
four_factor_global_day_hour_interaction WAPE = 0.6267
```

この結果は、day/hour 分割が小売データに合う導入であることを示す。一方で、interaction まで含む 4 成分版は通常予測では安定した改善を示さなかった。

したがって、次に確認すべきことは、同じ表現構造を residual target に移したときに、direct target よりも成分分解の効果が明確になるかである。

## 目的

目的は 3 つ。

1. Exp-26 の direct forecasting 結果と対応する residual target 側の最小比較を作る。
2. `global/local` reference と day/hour split の差が residual target で大きくなるか確認する。
3. 結果に応じて proposal の論調を調整する。

## 比較設計

direct target 側は 2-Exp-26 の結果を使う。

residual target 側は 2-Exp-27 で新しく実行する。

```text
r = y - b
b = series_mean
```

比較する model family は 2-Exp-26 と揃える。

| target | model | 実装上の variant |
|---|---|---|
| direct `y` | `global + local` | `paper_global_local` |
| direct `y` | `global + day + hour` | `four_factor_global_day_hour` |
| direct `y` | `global + day + hour + interaction` | `four_factor_global_day_hour_interaction` |
| residual `y - b` | `global + local` | `paper_global_local_residual` |
| residual `y - b` | `global + day + hour` | `four_factor_global_day_hour_residual` |
| residual `y - b` | `global + day + hour + interaction` | `four_factor_global_day_hour_interaction_residual` |

## なぜ `series_mean` residual に絞るか

この実験は、residual target が常に良いことを示すためではない。

まずは、これまでの実験で residual 構造が残ることが分かっている `series_mean` に絞り、direct target と residual target の対照を作る。

`same_hour_recent_mean` はすでに hour 構造を強く吸収するため、改善幅が小さく、hour component も解釈しにくいことが分かっている。これは後続で「baseline と residual structure に依存する」ことを示す条件として扱う。

## 成功条件

強い結果:

- residual target で `four_factor_global_day_hour` または `four_factor_global_day_hour_interaction` が `paper_global_local_residual` を明確に上回る。
- direct target での改善幅より residual target での改善幅が大きい。
- high residual top10 や corrected cell MAE で改善が出る。

この場合、proposal は次のように強められる。

```text
通常の売上全体を直接分解する場合、day/hour split の利点は小さい。
一方、series_mean baseline で水準成分を除いた residual では、day/hour/interaction の帰納バイアスがより明確に効く。
したがって、本研究の主対象は raw demand forecasting ではなく residual structure learning である。
```

弱い結果:

- residual target でも `paper_global_local_residual` が最も良い。
- day/hour/interaction の改善が direct target と同程度か、それ以下である。

この場合は、proposal を次のように抑える。

```text
残差なら常に成分分解が有利というわけではない。
有効性は baseline が何を取り除き、残差にどの構造が残るかに依存する。
本研究は、残差構造が残る条件を診断し、その条件下で解釈可能な補正を行う方法である。
```

## 実行コマンド

smoke:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-27_direct_vs_residual_bridge_smoke.json
```

FreshRetailNet:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-27_direct_vs_residual_bridge_freshretailnet.json
```

## 出力

```text
runs/2-Exp-27_direct_vs_residual_bridge_smoke/summary.csv
runs/2-Exp-27_direct_vs_residual_bridge_freshretailnet/summary.csv
```

各 variant 配下には次が出る。

```text
history.jsonl
metrics.json
hour_profile.csv
```

## 結果の読み方

まず、Exp-26 の direct target 結果を固定する。

| target | model | WAPE |
|---|---|---:|
| direct | `global + local` | 0.6233 |
| direct | `global + day + hour` | 0.6133 |
| direct | `global + day + hour + interaction` | 0.6267 |

次に、Exp-27 の residual target 結果で、`paper_global_local_residual` に対する改善幅を見る。

特に見る指標:

- `corrected_cell_mae`
- `high_residual_top10_corrected_mae`
- `residual_mae`
- `residual_r2`
- `component_ablation_*` が出る場合はその delta

direct target の改善幅が小さい一方で residual target の改善幅が大きければ、「残差成分から表現学習する方がよい」という論調を強められる。

逆に residual target でも改善が小さい場合は、「残差 target は必要条件ではなく、baseline によって残る residual structure が重要」と整理する。

## smoke 確認

smoke は実行済み。

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-27_direct_vs_residual_bridge_smoke.json
```

smoke 結果:

| model | residual MAE | corrected cell MAE | baseline cell MAE | high residual top10 corrected MAE | residual hour corr |
|---|---:|---:|---:|---:|---:|
| `paper_global_local_residual` | 1.5537 | 1.5537 | 1.5631 | 4.6472 | -0.0094 |
| `four_factor_global_day_hour_residual` | 1.5465 | 1.5465 | 1.5631 | 4.7719 | -0.4372 |
| `four_factor_global_day_hour_interaction_residual` | 1.5890 | 1.5890 | 1.5631 | 4.5673 | 0.8638 |

これは smoke 用の小規模 synthetic なので、性能差の解釈には使わない。確認したい点は、3 つの residual variant が同じ config で学習・評価でき、`corrected_cell_mae`、high residual 指標、hour profile 指標が出ることである。

## FreshRetailNet 結果

実行コマンド:

```bash
uv run decoupled-ts residual-experiment --config configs/2-Exp-27_direct_vs_residual_bridge_freshretailnet.json
```

結果:

| model | best epoch | residual MAE | residual R2 | baseline cell MAE | corrected cell MAE | corrected WAPE | corrected bias | top10 corrected MAE | hour corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_global_local_residual` | 10 | 0.0593 | 0.1152 | 0.0721 | 0.0593 | 0.9577 | -0.3966 | 0.2794 | 0.8937 |
| `four_factor_global_day_hour_residual` | 11 | 0.0671 | 0.0255 | 0.0721 | 0.0671 | 1.0838 | -0.1203 | 0.2907 | 0.8908 |
| `four_factor_global_day_hour_interaction_residual` | 10 | 0.0614 | 0.0007 | 0.0721 | 0.0614 | 0.9917 | -0.3898 | 0.2989 | 0.8797 |

`paper_global_local_residual` に対する差分:

| model | corrected cell MAE delta | corrected WAPE delta | top10 corrected MAE delta | residual R2 delta |
|---|---:|---:|---:|---:|
| `four_factor_global_day_hour_residual` | +0.0078 | +0.1261 | +0.0113 | -0.0897 |
| `four_factor_global_day_hour_interaction_residual` | +0.0021 | +0.0340 | +0.0196 | -0.1145 |

baseline に対する改善:

| model | baseline cell MAE | corrected cell MAE | MAE improvement | top10 baseline MAE | top10 corrected MAE | top10 improvement |
|---|---:|---:|---:|---:|---:|---:|
| `paper_global_local_residual` | 0.0721 | 0.0593 | 0.0129 | 0.2923 | 0.2794 | 0.0129 |
| `four_factor_global_day_hour_residual` | 0.0721 | 0.0671 | 0.0051 | 0.2923 | 0.2907 | 0.0016 |
| `four_factor_global_day_hour_interaction_residual` | 0.0721 | 0.0614 | 0.0108 | 0.2923 | 0.2989 | -0.0066 |

## FreshRetailNet の読み取り

2-Exp-27 は、当初期待した「residual target にすると four-factor latent が `global/local` を明確に上回る」という強い結果ではなかった。

最も良いのは `paper_global_local_residual` である。`series_mean` baseline の cell MAE 0.0721 に対し、0.0593 まで改善している。high residual top10 でも 0.2923 から 0.2794 に改善しており、`series_mean` residual を学習すること自体は有効である。

一方、`four_factor_global_day_hour_residual` は 0.0671、`four_factor_global_day_hour_interaction_residual` は 0.0614 で、どちらも `paper_global_local_residual` には届かなかった。特に day/hour のみの latent split は high residual top10 の改善も小さい。

したがって、2-Exp-27 からは次のように読むべきである。

```text
残差 target にすることは有効だった。
しかし、latent を global/day/hour/interaction に分けるだけでは、
元論準拠の global/local residual reference を明確に上回らなかった。
```

これは重要な失敗寄りの結果である。残差に移せば自動的に four-factor latent が効く、という論調は避ける必要がある。

## 成分利用の兆候

性能では `paper_global_local_residual` が最も良いが、`four_factor_global_day_hour_interaction_residual` には成分利用の兆候がある。

ablation は次の通り。

| model | zero global delta | zero day delta | zero hour delta | zero interaction delta |
|---|---:|---:|---:|---:|
| `paper_global_local_residual` | +0.0052 | - | - | - |
| `four_factor_global_day_hour_residual` | +0.0053 | -0.0033 | -0.0021 | - |
| `four_factor_global_day_hour_interaction_residual` | -0.0000 | +0.0159 | +0.0231 | +0.0006 |

`four_factor_global_day_hour_interaction_residual` では、day を消すと MAE が +0.0159、hour を消すと +0.0231 悪化する。これは、interaction 付きモデルの内部では day/hour 成分が補正に使われていることを示す。

ただし、最終的な corrected MAE では `paper_global_local_residual` より悪い。つまり、成分は使われているが、latent concat 型の decoder ではそれが全体性能に十分つながっていない。

この結果は、これまでの output decomposition 系の結果と整合する。

```text
潜在表現を分けるだけでは、decoder 内で情報が混ざる。
成分を解釈可能にし、補正性能にも結びつけるには、
出力そのものを global/day/hour/interaction に分ける設計が必要である。
```

## Exp-26 との比較

Exp-26 direct target:

| target | model | WAPE |
|---|---|---:|
| direct | `global + local` | 0.6233 |
| direct | `global + day + hour` | 0.6133 |
| direct | `global + day + hour + interaction` | 0.6267 |

Exp-27 residual target:

| target | model | corrected WAPE |
|---|---|---:|
| residual | `global + local` | 0.9577 |
| residual | `global + day + hour` | 1.0838 |
| residual | `global + day + hour + interaction` | 0.9917 |

WAPE は direct target と residual target でスケールが異なるため、絶対値を直接比較するべきではない。見るべきなのは、各 target 内での relative ordering である。

direct target では `global + day + hour` が `global + local` より小幅に良い。一方、residual target では `global + local` が最も良い。したがって、2-Exp-27 は「residual にすれば latent の four-factor split がより有利になる」という仮説を支持しない。

ただし、residual target では全モデルが baseline を補正している。特に `paper_global_local_residual` と interaction 付きモデルは baseline MAE を大きく下げた。つまり、支持されるのは次の主張である。

```text
series_mean baseline 後の residual には学習可能な構造が残る。
しかし、その構造を単に latent split で表すだけでは不十分であり、
出力分解や centering constraint が必要である。
```

## Proposal への反映

2-Exp-27 の結果を受けて、proposal は強めるのではなく、次のように精密化する。

避けるべき主張:

```text
残差 target にすると、global/day/hour/interaction latent split が global/local より明確に良くなる。
```

採用する主張:

```text
通常予測では day/hour split が小幅に効くが、interaction は不安定である。
series_mean baseline 後の residual には学習可能な構造が残り、baseline 補正は可能である。
しかし、潜在表現を細分化するだけでは global/local reference を安定して上回らない。
したがって、本研究の主張は latent split そのものではなく、
残差出力を global/day/hour/interaction 成分に分け、
centering などの制約で出力空間の意味を固定する点に置く。
```

この結果により、2-Exp-11 / 2-Exp-22 で示した output decomposition の重要性が強まる。2-Exp-27 は「latent split だけでは足りない」という bridge / negative result として扱う。
