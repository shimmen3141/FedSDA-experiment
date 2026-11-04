# Implementation Plan

- [ ] 1. 構成と挙動とtestを足す
  - 要求3.1〜3.3のtestを先に書き、REDを確かめてから、設定の項目（割り当ての方式・窓・整合の開始の新しい値、$\epsilon$・$\gamma$・温度のパラメータと $\epsilon$ の掃引の軸）、4構成、runtimeの分岐（表現部の共有と個人の分類層、診断用の分類層、クラスごとのMADDとDBSCAN、組ごとの平均、アンカーと対照の整合、欠けたクラスの規則、2クラスの2出力の分類層）、round数の掃引の値の許可、通信と計算量の計数、記録の表（保存形式の版を上げる）を足す。`CONFIGURATIONS_PROTECTED_WITHOUT_GOLDEN`と依存境界へ登録する。公式の式の抜粋`tests/refactoring/fedccfa_official_excerpt.py`（MIT Licenseの表記つき）を作る。
  - 完了条件: 新しいtest（公式の式との照合、元の条件での挙動の確認、小さい全体run）、構成・掃引・識別子・依存境界・スキーマのtest、FedEM・IFCA・Window・CFL・DriftSurf・FedDrift・Oracleのtest、基準版1〜16のgoldenの照合（Windows）、`tests/test_regression.py`が通る。Ruff・Pyright。新しい全体runのtestの所要時間を記録し、重いものは`slow`の印と`tests/conftest.py`の配り方へ登録する。
  - _Requirements: 1.1〜1.10, 2.1, 3.1〜3.4_
- [ ] 2. 文書と検証の記録
  - 要求4.1の文書（構成の一覧、`differences-from-feddrift.md`、`python -m tools.generate_option_docs`での設定参照の再生成）とnaming.mdの追記。関係するtestの結果をImplementation Notesに記録する。実装レビュー（独立CLI）と全pytestは主担当が行う。
  - _Requirements: 4.1_
  - _Depends:_ 1
- [ ] 3. 予備の実行と集計
  - commit・pushの後、主担当が内部で計画を確かめ、予備の実行（192 run）のコマンドを渡す。design.mdの集計と選ぶ手順の結果をユーザーに示す。開始と結果を実験の台帳に記録する。
  - _Requirements: 4.2_
  - _Depends:_ 2
- [ ] 4. $\epsilon$ の調整の実行と本規模の実行
  - 選んだ構成と $R$ で $\epsilon$ の調整の実行（480 run）、続いて本規模の実行（400 run）のコマンドを渡し、集計する。開始と結果を実験の台帳に記録する。
  - _Requirements: 4.2_
  - _Depends:_ 3

## Implementation Notes
