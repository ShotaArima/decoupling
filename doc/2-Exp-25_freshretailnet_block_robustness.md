# 2-Exp-25: FreshRetailNet Block Robustness

## 背景

2-Exp-24 では、FreshRetailNet の系列数を 2000 / 6000 / 12000 に増やしても、`series_mean` residual で改善が保たれることを確認した。

ただし、現在のデータ作成は `store_id, product_id` で並べた系列を先頭から順に使う。

そのため、2-Exp-24 で分かったことは「先頭から使う系列数を増やしても傾向が保たれる」ということであり、「系列の取り方を変えても傾向が保たれる」ことまでは確認できていない。

2-Exp-25 では、系列の開始位置をずらしたブロックを作り、同じ傾向が出るかを確認する。

## 目的

目的は 3 つ。

1. `series_mean` residual の改善が、先頭ブロックだけに依存していないか確認する。
2. `series_mean` residual では hour component が安定して残差の時間帯 profile に対応するか確認する。
3. `same_hour_recent_mean_d7` residual では、補正効果と解釈可能性が不安定であるという限界解釈が、別ブロックでも保たれるか確認する。

## 実装上の変更

`dataset.series_start_offset` を追加した。

これは、系列を作るときに、条件を満たす系列を先頭から何件スキップするかを表す。

例えば次の設定では、先頭 6000 系列をスキップし、その後の 6000 系列を train に使う。

```json
{
  "dataset": {
    "series_start_offset": 6000,
    "max_train_series": 6000,
    "max_eval_series": 1500
  }
}
```

cache file 名にも offset を含めるため、既存 cache と混ざらない。

## 比較条件

| scenario | baseline | start offset | train 系列上限 | eval 系列上限 |
|---|---|---:|---:|---:|
| `series_mean_block0_6k` | `series_mean` | 0 | 6000 | 1500 |
| `series_mean_block1_6k` | `series_mean` | 6000 | 6000 | 1500 |
| `series_mean_block2_6k` | `series_mean` | 12000 | 6000 | 1500 |
| `same_hour_recent_mean_d7_block0_6k` | `same_hour_recent_mean`, recent_days=7 | 0 | 6000 | 1500 |
| `same_hour_recent_mean_d7_block1_6k` | `same_hour_recent_mean`, recent_days=7 | 6000 | 6000 | 1500 |
| `same_hour_recent_mean_d7_block2_6k` | `same_hour_recent_mean`, recent_days=7 | 12000 | 6000 | 1500 |

seed は 3 種類にする。

```text
17, 23, 31
```

これは厳密な交差検証ではないが、系列選択の偏りを見るための block robustness check である。

## 比較 model

| model | 目的 |
|---|---|
| `mae_grid_reference` | MAE 最小化を重視した補正 |
| `bias_constrained_001` | bias を抑えた補正 |

2-Exp-24 で `centered_raw` は calibration の有無を見る役割だった。2-Exp-25 では論文上の主比較に絞るため、calibration 済みの 2 条件にする。

## 良い結果

`series_mean` residual について、次が複数 block で保たれるなら良い。

- corrected MAE または calibrated corrected MAE が baseline MAE より低い。
- high residual top10 が baseline より改善する。
- hour component residual profile corr が正で高い。
- hour component を消したときの MAE 悪化が明確に出る。

この場合、`series_mean` residual の結果は、先頭系列だけに依存しないと説明できる。

## 悪い結果

次のような結果なら、主張を弱める必要がある。

- `series_mean_block0_6k` では改善するが、block1 / block2 で改善しない。
- hour corr が block ごとに大きく変わる。
- high residual top10 の改善が block ごとに消える。

この場合、FreshRetailNet での改善は「系列選択に依存する」として limitation に明記する。

## 実行コマンド

smoke:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-25_freshretailnet_block_robustness_smoke.json
```

本実験:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-25_freshretailnet_block_robustness.json
```

## 出力

```text
runs/2-Exp-25_freshretailnet_block_robustness/all_results.csv
runs/2-Exp-25_freshretailnet_block_robustness/aggregate.csv
runs/2-Exp-25_freshretailnet_block_robustness/summary.json
```

## 論文での使い方

良い結果なら、FreshRetailNet 実験の前提を次のように補強できる。

```text
FreshRetailNet では全系列交差検証ではなく公式 split に基づく評価を用いたが、
系列ブロックを変えた感度分析でも series_mean residual の改善と hour 成分の対応は保たれた。
```

悪い結果なら、次のように limitation として使う。

```text
実データでの改善は residual target だけでなく対象系列にも依存する。
したがって、提案法は無条件な汎用予測器ではなく、
残差に再現可能な時間帯構造が残る領域で有効な補正手法である。
```

## 結果

実行コマンド:

```bash
uv run decoupled-ts residual-sweep --config configs/2-Exp-25_freshretailnet_block_robustness.json
```

結果は 3 seed 平均で読む。

| scenario | model | baseline MAE | corrected MAE | calibrated MAE | top10 baseline | top10 corrected | top10 calibrated | bias | calibrated bias | hour corr | hour ablation delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `series_mean_block0_6k` | `mae_grid_reference` | 0.0697 | 0.0507 | 0.0508 | 0.2788 | 0.2062 | 0.2098 | -0.2544 | -0.2553 | 0.9957 | 0.0101 |
| `series_mean_block0_6k` | `bias_constrained_001` | 0.0697 | 0.0510 | 0.0545 | 0.2788 | 0.2132 | 0.1988 | -0.2511 | -0.0143 | 0.9898 | 0.0099 |
| `series_mean_block1_6k` | `mae_grid_reference` | 0.0671 | 0.0498 | 0.0494 | 0.2581 | 0.1950 | 0.1986 | -0.2208 | -0.2896 | 0.9766 | 0.0096 |
| `series_mean_block1_6k` | `bias_constrained_001` | 0.0671 | 0.0490 | 0.0522 | 0.2581 | 0.1972 | 0.1796 | -0.3167 | -0.0069 | 0.9812 | 0.0089 |
| `series_mean_block2_6k` | `mae_grid_reference` | 0.0693 | 0.0518 | 0.0507 | 0.2617 | 0.1877 | 0.1918 | -0.1891 | -0.2594 | 0.9754 | 0.0102 |
| `series_mean_block2_6k` | `bias_constrained_001` | 0.0693 | 0.0514 | 0.0546 | 0.2617 | 0.1966 | 0.1788 | -0.2656 | -0.0057 | 0.8865 | 0.0096 |
| `same_hour_recent_mean_d7_block0_6k` | `mae_grid_reference` | 0.0580 | 0.0563 | 0.0562 | 0.2534 | 0.2416 | 0.2426 | -0.1587 | -0.1699 | -0.8870 | 0.0005 |
| `same_hour_recent_mean_d7_block0_6k` | `bias_constrained_001` | 0.0580 | 0.0564 | 0.0582 | 0.2534 | 0.2384 | 0.2453 | -0.1260 | 0.0000 | -0.8767 | 0.0002 |
| `same_hour_recent_mean_d7_block1_6k` | `mae_grid_reference` | 0.0555 | 0.0545 | 0.0543 | 0.2436 | 0.2346 | 0.2367 | -0.1615 | -0.1828 | -0.8694 | 0.0003 |
| `same_hour_recent_mean_d7_block1_6k` | `bias_constrained_001` | 0.0555 | 0.0537 | 0.0558 | 0.2436 | 0.2296 | 0.2335 | -0.1573 | -0.0000 | -0.8467 | 0.0004 |
| `same_hour_recent_mean_d7_block2_6k` | `mae_grid_reference` | 0.0569 | 0.0561 | 0.0554 | 0.2436 | 0.2350 | 0.2352 | -0.2105 | -0.2007 | -0.8421 | 0.0004 |
| `same_hour_recent_mean_d7_block2_6k` | `bias_constrained_001` | 0.0569 | 0.0557 | 0.0576 | 0.2436 | 0.2358 | 0.2406 | -0.1917 | 0.0000 | -0.8085 | 0.0007 |

## 読み取り

`series_mean` residual では、block0 / block1 / block2 のすべてで baseline を明確に上回った。

MAE 改善幅は次の通り。

| scenario | `mae_grid_reference` 改善 | `bias_constrained_001` 改善 |
|---|---:|---:|
| `series_mean_block0_6k` | 0.0190 | 0.0186 |
| `series_mean_block1_6k` | 0.0173 | 0.0181 |
| `series_mean_block2_6k` | 0.0175 | 0.0179 |

高残差上位 10% でも全 block で改善した。特に `bias_constrained_001` の calibrated top10 は、block0 で 0.1988、block1 で 0.1796、block2 で 0.1788 まで下がった。これは、bias を抑えた calibration が全体 MAE ではやや不利でも、外れケース補正では有利になりやすいことを示している。

hour component の対応も保たれた。`mae_grid_reference` では hour corr が 0.9754〜0.9957、`bias_constrained_001` でも 0.8865〜0.9898 である。hour component を消したときの MAE 悪化も 0.0089〜0.0102 程度で、時間帯成分が実際に補正に使われている。

一方、`same_hour_recent_mean_d7` residual では、MAE 改善は 0.0008〜0.0017 程度にとどまった。さらに hour corr はすべての block で負である。したがって、この residual target では「予測補正としてわずかに効く場合はあるが、hour component が残差の時間帯構造を素直に説明している」とは言いにくい。

## 結論

2-Exp-25 により、`series_mean` residual の改善は先頭系列ブロックだけに依存しないことが確認できた。

FreshRetailNet 実験は全系列交差検証ではないが、少なくとも `store_id, product_id` で並べた 3 つの異なる 6000 系列ブロックにおいて、同じ傾向が再現した。

論文では次のように整理できる。

```text
FreshRetailNet では公式 split に基づく評価を用いた。
全系列交差検証ではないが、系列ブロックを変えた感度分析でも、
series_mean residual の改善と hour 成分の対応は保たれた。
```

同時に、次の限界も明記する。

```text
same-hour recent mean のような強い基準値を用いると、
残差には時間帯構造が残りにくく、成分分解の解釈可能性は低下する。
```
