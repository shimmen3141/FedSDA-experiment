# 実装タスク

既存Python/NumPy/torch/pytestと設定・数値予測部品を利用し、環境導入は不要。全段階と命名承認はspec.jsonが正本。順次TDD→Lunaレビュー→主担当再検証→切り戻せる単位のcommitで実行する。

- [x] 1. 単一有界損失のe-SR状態と観測結果を移植する
  - baseline・alpha・保持上限・賭け率を明示入力とし、候補追加・古い候補除去・旧float64対数合成・同率候補・resetを実装する。
  - 結果と診断copyを分け、不正条件・損失・reset入力で全状態が不変となる。
  - 全候補capital・番号・対数e値・警報・幅・計数が旧検出器と各観測後に完全一致する対象検証が通る。
  - _Boundary: 単一e-SR数値状態_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.1, 4.2, 4.4, 5.1_

- [x] 2. 全体・正解クラス系列の混合とglobal位置対応を移植する
  - 既存alpha/監視対象と明示baselineを使い、class初出時のbaseline固定・成分更新・固定配分の旧順混合・警報・候補位置を実装する。
  - 全入力を更新前に検査し、reset・frozen snapshot・観測delta計数を公開する。
  - 遅延baseline・class局所変化・非連続class位置・保持上限・未観測class・成分同率・resetを旧ClassESR実メソッドに完全一致させるテストが通る。
  - _Boundary: 全体・正解classの混合/位置状態_
  - _Depends: 1_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.2_

- [ ] 3. 観測後損失へ接続し依存境界と全回帰を検証する
  - モデル別観測後損失のpublic APIから指定現行モデルの損失だけを監視へ渡し、予測重み・モデル状態を監視へ持ち込まない。
  - NumPyを単一exact moduleに限定し、混合側は同機能kernel/設定以外の依存を拒否する。全src AST走査と禁止依存注入検査が通る。
  - public keyword契約・snapshotと別実体独立・グローバル乱数不変を検証する。
  - 全refactoring・schema・旧11/最終3goldenを含む全tests・旧moduleをimportしない独立smokeが通り、旧production/golden差分なし、20条件・部分移植範囲・blockedなしの完成証拠が揃う。
  - _Boundary: 明示損失接続・監視・依存検査の統合_
  - _Depends: 1, 2_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4_

