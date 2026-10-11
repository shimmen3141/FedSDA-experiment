# Research & Design Decisions

## Summary

- Feature: `feddrift-detection-batch-assessment`。既存learningへの拡張。
- 旧の検出は、全モデルの有界損失をバッチ平均し、最小値の増加を比較する。順序と等号の扱いを維持する。
- 全体runの枠はFedSDAの終端操作を含む。FedDriftへの接続を、本specの数値移植と混ぜない。

## Research Log

### 旧の数値と時系列

- `federated_drift_experiment/clients/feddrift.py::_detect_and_assign`は、保有dictの順で評価し、`loss < min_loss`だけで最小を更新する。現在のモデルを優先する同率処理はない。
- ドリフトは`min_loss > last_min_loss + distance_threshold`。初回は検出しない。`flush`の部分バッチも同じメソッドを使う。
- `models.py::SimpleMLP.per_sample_error`の二値・多クラスの計算は、既存の`classifier_bounded_loss_evaluation.py`と一致する。
- 旧は判定後にモデル生成・割当・評価用標本抽出を行い、乱数を消費する。本specはその直前までを所有する。

## Design Decisions

- `agent-handoff.md`の「数値やアルゴリズムの一致が繊細な部分は小さいspecへ切り出してよい」に従う。判定後の処理を含むclientの移植と全体run照合は後続に残す。この境界を全体FedDriftの完了とは扱わない。
- 学習bindingではなくIDと分類器の保有順tupleを受ける。optimizerと状態ownerは、この計算に必要ない。
- 同率選択・Tensor平均・閾値比較を独自に一般化せず、1つの関数にまとめる。結果のためだけのmodule分割はしない。
- 利用skillはkiro-spec-init・requirements・design・tasks・impl。段階承認は共通引継ぎ手順の委任、task-graph sanityはinline（子エージェントは使わない）。既存の固定依存だけを使うため外部調査・新しいライブラリは不要。

## Risks & Mitigations

- 確率平均をPythonで合計すると末尾が変わるため、実旧と同じTensor平均を使う。
- 後続clientでは、初回・ドリフト時・モデル切替時の前回値の更新と評価データ抽出の乱数順を別途照合する。
- 新の不正入力はforward前に拒否する。旧の契約外入力の挙動を再現することは要求しない。
