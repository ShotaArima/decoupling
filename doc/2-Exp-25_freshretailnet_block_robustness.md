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
