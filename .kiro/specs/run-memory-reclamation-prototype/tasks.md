# Implementation Plan

task 1開始前に仕様書群の独立レビューを通過させる。下のtask 3のレビューは全実装後の実装レビューであり、この事前ゲートを代替しない。

- [x] 1. 独立した実行境界と回収の試作を検証する
  - 専用checksを先に用意し、計画純読取・事前拒否、同期/回収順序、例外復元、control非回収を確認する。
  - 既存CLI保存を用いる2armの新規process逐次起動・identity・環境・RSS/時間・最終params記録を作る。
  - fake依存とWindowsの専用checkで、学習フローへ影響を持ち込まず境界が動くことを確認する。
  - _Requirements: 1, 2_
- [x] 2. 全保存値の照合と実NNの接続を検証する
  - 指標/配列/params/identity/未完了の異常対照を先に作り、誤ってPASSしないことを確認する。
  - Linuxの50/client・2runで設定/全指標/全133配列bytes/最終paramsの一致と正常終了を確認する。
  - 既存src・共通設定の差分なし、専用品質検査、Python3.10文法と計画2runを確認する。
  - _Depends: 1_
  - _Requirements: 2, 3_
- [x] 3. サーバ検証の手順と独立レビューを完了する
  - 全実装・配布手段の検証後の独立実装レビュー指摘を処理し、git pullまたはZIP転送・踏み台経由の送受信・tmux実行を文書化する。
  - 効果/時間/同一性の判断はサーバ返却後であることを明示し、完了した検証と未実行を分けて引き継ぐ。
  - _Depends: 2_
  - _Requirements: 1, 2, 3_

## Implementation Notes

仕様レビューはLuna highでPASS、Major0/Minor1。task 3のレビュー時期の明示を採用し、Minor修正のみなので再レビューは行わない。副作用境界と数値不変条件に従い実装する。

- 初期checks8件は未実装のためRED。その後、専用checks23件が固定Windows環境で成功（JUnit: `../../venv/refactoring-tests/memory-reclamation-unit.xml`）。標準権限でのtmp操作拒否は既知の制限で、主担当の許可済み昇格実行で検証。恒久的な権限変更なし。
- Windowsのread-only計画は2armを表示し、出力ディレクトリを作成していない。Python3.10文法、isolated Ruff/format、`spec_checks.py names`成功。通常pytest/CIへの登録なし。
- Linux50/clientのcontrol/reclaimはexit0、完全設定/全指標/全133配列bytes/最終params/環境が一致。同期末の生存状態で最大412.25 MiBを回収、ピーク差は0.26 MiBで接続確認以上には一般化しない。`results/fedsda-memory-diagnosis/reclamation-smoke50/`が証拠。実装の整形・import順調整前の記録であり、そのrunner hashを保持している。
- 整形後runnerで1000/client・2runもexit0、設定/全指標/全133配列bytes/最終params/環境が一致。20同期で回収、最大同期末RSS低下440.88 MiB、ピーク1173.98→1114.03 MiB、wall145.16→152.00秒、回収呼出し累積0.466秒。手元で約5.1%/4.7%の観測だがサーバの効果/性能率には一般化しない。`results/fedsda-memory-diagnosis/reclamation-local1000/`。
- 配布用ZIPはsrcではなくgit管理外のresultsに展開し、同一runner byte/hash・動的なroot検出・計画純読取を確認する。実装レビュー完了前に本規模のサーバ効果や既定採用を主張しない。
- 独立実装レビューはLuna high、session `01a15117-74c4-7711-a302-fd8c19a52243`、PASS Major0/Minor1。133配列の件数も必要という指摘を採用し、固定件数ゲートと両側同時欠落の拒否テストを追加。最終専用checks24件成功。Minorのため再レビューは行わない。
- レビュワーはRuff/format・名前/progress・Python3.10文法を独立確認。専用pytestは標準Temp/指定basetempとも権限で停止し、当時の主担当JUnit23件・失敗0を照合。修正後の24件は主担当の実測。全pytest/fresh/NN学習は独立未実行。許可済みの主担当証拠による判定を適用。一時的に作ったレビュー領域は主担当が空であることを確認して非再帰で片付けた。
- 最終配布runnerの`results/fedsda-memory-diagnosis/reclamation-final-package-smoke50/`もexit0、全133配列/指標/設定/最終params/環境一致。元の1000/clientは件数ゲート強化前runnerのidentityを保持し、最終の照合関数で再読取して成功を確認した。ゲート修正によって学習部分を変更したと扱わない。
- ZIPは`results/fedsda-memory-diagnosis/uranium-live-reclamation-20261019.zip`。元srcのhashは不変、通常pytest/CIへの登録・製品の既定採用なし。ユーザーによるuraniumの5000/client検証は未実行。転送/実行/返却手順は`docs/experiments/run-memory-reclamation-prototype.md`。
