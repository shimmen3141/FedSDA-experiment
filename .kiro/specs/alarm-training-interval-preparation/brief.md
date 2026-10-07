# 警報時の学習区間の準備 — 境界決定

日付: 2026-10-08。固定旧基準: `748c3aa`。直前の完了specは`alarm-change-interval-resolution`。

## Boundary Candidatesと採用理由

残りの`FedSDAClient._resolve_drift`は次の2責務へ分ける。

1. **今回: 保留位置から前区間と変化区間を取り出し、前区間の評価標本保存→現行モデルへの吸収を済ませ、変化区間を返す。** 区間分割と前区間更新の順序、位置とpayloadの対応、Python乱数が主要な検証点。既存の分割・評価保存・吸収を再利用し、それぞれの計算を再実装しない。
2. **後続: 警報全体の制御と後始末。** active sessionの有無、最小区間件数、変化区間の解決、検出器reset、FIFO消費、適応イベント・切替位置・再利用計数・通知・session保持を組み立てる。今回の結果を受け取り、十分な件数のときだけ既存の変化区間解決を呼ぶ。実際にspecを作る時点でさらに分割が必要か判断する。

前区間の吸収は現行モデルの履歴損失を変えるため、分割だけを独立して完了扱いにすると、後続が評価前の更新順を見落としやすい。この2処理を同じ準備の責務とする。一方、FIFOを消費するかは警報の分岐によって異なるので、今回には含めない。

## 根拠

- 旧`clients/fedsda.py::_resolve_drift`: active sessionありの分岐が先。通常経路は末尾`min(FIFO件数, 推定span)`を変化区間にし、前区間の評価保存と吸収を先に実行する。最小件数未満でも前区間処理は実行済みで、FIFOは残る（LEGACY-002）。
- `PendingTrainingAssignmentBuffer.get_change_interval_partition`: 推定spanが正の整数なら位置の区間を返す。capacity+1の警報時点も扱え、FIFOを消費しない。
- `ModelEvaluationSampleStore.sample_and_append_model_evaluation_samples`: 明示した`random.Random`を使い、正規IDに抽出保存する。一時IDには保存せず乱数も消費しない。
- `absorb_assigned_training_samples_into_held_model`: 標本・診断用概念IDを受け、既存の吸収と同じ統計・標本・計数更新を行う。
- `ClientObservedStream`は現在2特徴・二値のscalar recordであり、多クラスTensor標本の汎用所有者として拡張しない。今回の設計で、呼出側が明示位置・観測済み標本・概念IDを結び付けて供給する境界を定める。明示位置列を保留位置列と照合し、payloadの意味上の正しい位置対応は供給側が保証する。真の概念IDは診断専用で、分割・帰属選択には用いない。

## 隣接契約

active sessionがない通常経路専用。呼出側は警報時点の推定spanと観測済み標本を供給し、準備の完了後に件数判定と区間解決を行う。検出器からspanを推定する処理は既存検出の責務。LEGACYの修正・アルゴリズム改善を移植へ混ぜない。

## 証拠の予定

旧`_resolve_drift`の前区間処理が実際に通る対照を用いる。区間評価まで接続するテストでは吸収後の履歴統計が使われることも確認する。旧global Python乱数と新明示Randomを同じstateから実行し、保存された標本順と最終stateを照合する。torch/NumPy乱数、モデル・optimizer、FIFO・帰属IDの不変も別に確認する。
