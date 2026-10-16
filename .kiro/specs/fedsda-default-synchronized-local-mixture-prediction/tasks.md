# Implementation Plan

- [x] 1. 基準版1〜5と旧と比べる入口で、予測に使うパラメータとTを明示する（既定は、まだ変えない）
  - `build_case_settings`が、版1〜5で予測に使うパラメータ（局所）とT（30）を明示する。旧と比べる入口（`use_legacy_compatible_policies`）が、2つを明示し、入口の検査（`assert_run_client_uses_legacy_compatible_policies`）が確かめる。入口の検査が混ぜる予測の設定を拒否するtestを先に書く。
  - この時点で、全部のtestが値を変えずに通る（既定と同じ値を明示しただけ）。
  - _Requirements: 2.1, 2.2_

- [x] 2. 最終構成の既定を替え、派生の構成を合わせる
  - 構成の既定のtest（全FedSDA構成の予測に使うパラメータとT、全構成の組立てと小さい実行、コマンドの引数の一覧）と、`fedsda-local-parameter-prediction`の専用のtestを先に書き、REDを確かめる。
  - 既定に依る既存のtestは、設計の「既定に依る既存のtestの扱い」の表のとおり合わせる。最終構成の2つの既定、`fedsda-single-model-prediction`の明示、`fedsda-local-parameter-prediction`の追加（構成の一覧、goldenにしない構成の登録）、混ぜる予測の構成の説明の文。既定に依るtestを、失敗したものから合わせる（実旧と比べるtestの期待値は変えない）。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1_

- [x] 3. 基準版6を足す
  - 版6の定義と、版5→6の差の説明のtest（設計の1〜5と、予測の側の一覧）を足す。sourceとtestをcommitしてから（この中間のcommitでは、版6のgoldenがまだ無いので、両方の基準環境のgoldenが揃うことを確かめるtest——`test_committed_goldens_cover_both_reference_environments`ほか——が失敗する。生成の前にsourceのcommitが要り、生成の選択肢にも版6が要るので避けられない。前の版——版5——と同じ扱いで、生成とgoldenの追加を同じ作業の中で続けて行う）、`--baseline-version 6`で、Windowsの基準環境とWSLで生成して足し、照合する。現在の版（`BASELINE_VERSION`）を6へ替えるのは、goldenを置いた後。
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 4. 文書を更新する
  - 実験の入口の文書（構成の表、容量とTの節）、基準の文書（版6）、研究バックログの8（既定の判断）、最終提案構成の文書（固定設定）、再開案内、完了specの一覧。
  - _Requirements: 4.1_

- [x] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSL（goldenの版1〜6と、このspecのtest）。
  - _Requirements: 全部_

## Implementation Notes
- 2026-10-17引継ぎ: Task1・2の既存変更を保持し、Task3から続行。ClaudeのログはTask1 321件成功、Task2 3374件成功＋追加修正の単独成功を報告するが、checkbox未更新だった。主担当が再検証中。
- Task3の設定契約testは版6未登録でRED（1 failed）、版6登録後の設定/予測側差分testは2 passed（Windows、167秒）。版5と同じ19ケースを使い、学習・通信・全体パラメータ・coverageの差は許容しない。
- 生成前の独立コードレビューはユーザー指定のLuna highでPASS。数値testは主担当が実施し、最終独立レビューはgoldenと文書が揃った後に行う。初回CLI入力の文字化けを直し、ASCII依頼で再実行した（spec.json）。
- Ruff lint/formatとpip check成功。Pyrightはsandbox内の相対Python指定で依存解決に失敗したため、固定venv絶対パスと必要なプロセス権限で再実行した。
- 以下の中間commitではsrcの来歴を固定し、版6存在検査と版定数一致検査はまだ未成立。設計どおり、新golden設置後に現在版と実行来歴版を6へ上げて解消する。

- 引継ぎ時点のTask1・2と版1〜5の回帰を固定Windows環境で再検証し、453 passed（JUnit `../../venv/refactoring-tests/default-mixture-resume.xml`、748秒）。この実行に後から追加した版6のtestは含まれない（別途2 passed）。生成元は中間commit `5e85456`。

- 基準版6は生成元`5e85456`からWindows/WSLそれぞれ独立2processの全snapshot一致で生成成功。両環境の19ケースとMNIST hash/環境一致、版5と同じ最終パラメータ/coverageを確認してv6へ新規設置した。旧版は上書きしていない。設置後に現在版と実行来歴版を6へ揃えた（既存契約を設計へ明記）。

- 最終実装レビュー（Luna high）初回はP1: 研究バックログの採用判断の後に古い未決の推奨が残る。過去の検討と明示して、限定再確認PASS（各sessionはspec.json）。
- 主担当最終検証: Windows全pytest 14370 passed / 8 skipped / 失敗0（462秒、JUnit `../../venv/refactoring-tests/default-mixture-full.xml`）。旧11/最終3goldenを扱う2回帰testも成功。skipは既定Windows側の代替回帰・Linux専用・POSIX bash専用。WSLの版1〜6と対象契約は271 passed。
- 独立検証（Luna high）: 対象/依存境界3172 passed、Ruff成功、Git状態不変。freshはsandbox内のTemporaryDirectory配下のNPZ書込みと走査で権限エラー。主担当の未変更共用scriptは16フローで成功した。全pytestは既存決定どおり主担当実測のみ。freshにも環境制約時の証拠照合を適用するかユーザー確認中。
- 品質検査はRuff lint/format、Pyright 0 errors、pip check成功。固定旧と版1〜5goldenの差分なし。実行時sourceのLF hashは`25a7bc722dfb917795d2dba344329245e8b92e54f8ec873c5b0b811f3f01035f`（480パス、同じ計算法でGit commit後も照合する）。

- 2026-10-17ユーザー決定: 権限により独立実行不能なfresh smokeは、主担当の成功証拠をレビュワーが照合して判定する。共通手順へ追記した。対象3172件とRuffは独立成功、fresh16フローと全pytestは主担当実測であり、独立再実行済みとはしない。最終証拠照合を依頼する。

- 最終証拠照合（Luna high）PASS。480パスの作業ツリーsource hash、全pytest/独立対象/WSLのJUnit集計、旧回帰と版6全19ケース、fresh成功ログ、固定旧/版1〜5不変を独立確認した。環境例外を適用してGO。全pytest・WSL・freshをレビュワーが再実行したとは記録しない。
- 表現上の補正: 最小差が−0.03ポイントなのに「下回らない/T30以上」としていた要求の背景と研究バックログを「ほぼ同等以上」へ直した。採用条件・実装・数値は不変で、文言のみのため再レビューは不要と判断。
- 最終提案の入口、容量/T、旧比較、版6、研究判断、再開案内と完了spec一覧を更新した。未採用の自動調整・名称変更・機能削減は行わず、研究室サーバ用goldenも追加していない。最終commitのsource hash同一性とpushの結果はGitで確認する。
