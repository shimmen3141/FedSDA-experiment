# 実装タスク

全段階承認と命名revisionはspec.jsonが正本。共有数値moduleを順番にTDD→Lunaレビュー→主担当再検証→commitする。
既存Python/torch/pytestと上流重み部品を使い、環境の新規導入は不要。task 1～2は対象機能・既存設定の段階検証、task 3で全src依存検査と全goldenをgateにする。

- [x] 1. モデル出力の確率化と混合用重みの正規化を用意する
  - 明示クラス数、非空・共通標本数、CPU float32形状、有限値・IDを検査する。
  - 二値再sigmoidなし、多クラスのクラス軸確率化、通常昇順加算による重み正規化を実装する。
  - 独立storage/no_grad・入力不変と旧基準同値の対象テストが通る。package入口は説明のみとする。
  - _Boundary: Classification prediction calculations_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.4, 5.1, 5.2_

- [x] 2. 混合・クラス判定と観測後モデル損失を移植する
  - 同ID集合とモデル別確率を検査し、既に正規化された重みを昇順の積・逐次加算へそのまま使う。
  - 混合後スコアの値を補正せず、二値閾値・多クラス同率、ラベル観測後float32平均損失を実装する。
  - 閾値隣接・同率・混合丸めの自己拒否防止・不正ラベル・返却独立性で旧基準に完全一致するテストが通る。
  - _Boundary: Classification prediction calculations_
  - _Depends: 1_
  - _Requirements: 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2_

- [ ] 3. 重み状態へ接続して依存境界・既存回帰を検証する
  - public APIだけで予測前取得→正規化→混合→ラベル観測→同snapshotで重み更新をつなぎ、各標本後に旧経路と照合する。
  - exact数値moduleのtorch依存だけを許し、上位・旧module・NumPy・method/controller依存を注入テストで拒否する。
  - 新src全走査、全refactoring、schema・tests/test_regression.py・tests/test_proposed_regression.pyを含む全testsと独立プロセスsmokeが通る。
  - 旧production/golden差分なし、全20条件の証拠とroadmapの部分移植範囲が一致し、blockedのない完成記録を残す。
  - _Boundary: 数値部品・重み状態・依存検査の明示統合_
  - _Depends: 1, 2_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4_
