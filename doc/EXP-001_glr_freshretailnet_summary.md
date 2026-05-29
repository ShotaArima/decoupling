# EXP-001: GLR FreshRetailNet 実験結果サマリー

## 実験目的
以下 3 タスクに対する現在の実験結果を整理します。

- Downstream Prediction
- Subgroup Identification
- Forecasting

## 実験結果（数値）

| タスク | 指標 | 値 |
|---|---:|---:|
| Downstream Prediction | MAE | 1.3104 |
| Downstream Prediction | MSE | 12.3733 |
| Subgroup Identification | Accuracy | 0.51622 |
| Forecasting | MAE | 1.9151 |
| Forecasting | MSE | 12.2542 |

> 元データ（生値）
>
> - downstream_prediction: mae=1.310418725013733, mse=12.373334884643555
> - subgroup_identification: accuracy=0.51622
> - forecasting: mae=1.915102243423462, mse=12.254233360290527

## 簡易考察

1. **Downstream Prediction**
   - MAE は 1.31 程度で、3 タスク内では誤差が比較的小さい結果でした。
   - MSE が 12.37 であるため、一部のサンプルで大きめの誤差が出ている可能性があります。

2. **Subgroup Identification**
   - Accuracy は **0.51622**（約 51.6%）でした。
   - 2 クラス分類を想定した場合、ランダム水準（約 50%）に近く、識別性能の改善余地が大きいです。

3. **Forecasting**
   - MAE は 1.92 程度で、Downstream Prediction より誤差が大きくなっています。
   - MSE は 12.25 と Downstream Prediction と同程度で、平均的誤差は大きい一方で、外れ値による影響度は近い水準です。

## 次アクション案

- **比較軸の明確化**: ベースライン（Naive/既存手法）と同一データ分割で比較。
- **閾値の再設計**: Subgroup Identification のクラス不均衡を確認し、しきい値最適化や F1/AUROC も併記。
- **誤差分解**: MAE と MSE の差分をサンプル単位で分析し、外れ値・特定期間・特定カテゴリの寄与を確認。
- **再現性確保**: 乱数 seed、学習設定、評価時設定を固定し、複数 seed の平均と分散を記録。

## 関連実験

小売向けに `z_global / z_day / z_hour / z_interaction` へ拡張した実験は、`EXP-002` として別ドキュメントに分離しています。
