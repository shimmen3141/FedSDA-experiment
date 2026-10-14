# Implementation Plan

- [x] 1. 版の差の事象`alarm_wait_situation_count`を足し、基準版4の警報の扱いを明示する（既定は、まだ変えない）
  - `capture_case_and_count_divergence_events`で、標本1件の処理のmoduleの`handle_alarm_occurrence`を包み、待つ扱いなら警報を抑える場面（設計の条件）の回数を数える（このtaskでは、e-SRの検出器のFedSDAのケースで数える。次の版で待つ扱いを使うケースへの限定は、版5を定義するtask 3で行う）。「版の後に、既定の値が変わった項目」の表へ、版4の警報の扱いを足し、`build_case_settings`がその経路を扱う。
  - この時点で、全部のtestが値を変えずに通る。版4の扱いで、版4の全ケースの事象の回数を確かめ、記録する。1以上のケースが無ければ、ユーザーへ確認する。
  - _Requirements: 2.2, 3.2_

- [x] 2. 最終構成の既定を案Yにし、ADWINの2構成の警報を即時に明示する
  - 構成の既定のtest（11構成の登録と警報の値、ADWINの2構成は即時、全構成の組立てが例外を出さない、掃引のコマンドの引数の一覧が変わらない）を先に書き、REDを確かめる。既定に依るtestの候補: 掃引のrunの識別子の期待値、最終構成の既定の照合、構成の既定のtest、共用のfresh scriptの方針の名前、待つ扱いのspecの「既定は即時」のtest、`test_fedsda_configuration_variants.py`のADWINへ替える補助関数と構造の比較（既定の待つ扱いを引き継いだまま、ADWINへ替えると拒否される）。
  - 最終構成の2つの既定と、ADWINの2構成を替える。既定に依るtestを、失敗したものから合わせる（実旧と比べるtestの期待値は変えない。失敗したら、旧と同じ扱いの明示の漏れを直す）。依存境界の許可の名前を入れ替える。sourceの既定の注記を直す。
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1_

- [x] 3. 基準版5を足す
  - 版5の定義と、版4→5の差の説明のtest（`(4, 5)`。名前の無いケースの検査を含む。ADWINの2構成が同じ値であること）を足す。版5で必須の経路が通らないケースがあれば、版5だけに、通るseedのケースを足す。
  - sourceとtestをcommitしてから（この中間のcommitでは、版5のgoldenがまだ無いので、版5のgoldenの存在を確かめるtestが失敗する。生成の前にsourceのcommitが要るので、避けられない。前の版のときと同じ）、`--baseline-version 5`で、Windowsの基準環境とWSLで生成して足し、照合する。現在の版（`BASELINE_VERSION`）を5へ替えるのは、goldenを置いた後。
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 4. 文書を更新する
  - 実験の入口の文書、基準の文書、IMPROVE-006の記録と一覧、LEGACY-016の記録、再開案内、比較scriptの説明の文。
  - _Requirements: 4.1_

- [ ] 5. 実装レビューと最終検証
  - 独立CLIの実装レビュー、指摘の反映、全pytest・Ruff・Pyright・WSLの対象testを通す。
  - _Requirements: 全部_

## Implementation Notes
- 仕様レビュー1回目（独立CLI、Claude Haiku 5.5、effort medium、session `832049c9-86f2-4fb3-a04d-2efe0a8ee7b9`、対象は基点`a30d8bd`の上の未commitの仕様書群。選んだ理由: 状態と分岐の論証なので、Haiku）: `SPECIFICATION: CHANGES REQUIRED`（Major 1、Minor 2）。採否（全部採用）:
  - M1（版の差の事象を、短い警報の応答の種類で数えると、候補検証を保持している間の場面を数え漏れる）: 数える場所を、待つ扱いの分岐の条件そのものへ改めた（標本1件の処理のmoduleの`handle_alarm_occurrence`を包み、引数から同じ式で計算する）。キーを`alarm_wait_situation_count`へ改めた。
  - m2（構成の数）、m3（版5の必須の経路の分岐と、seedの探し方）: 要求・設計へ書いた。
- 仕様レビュー2回目（修正の再確認。同じ担当・effort、session `f6bca78d-be69-4119-b90b-d4f25e664e80`）: 1回目のMajorは解消。新しいMajor 1、Minor 5で`CHANGES REQUIRED`。採否（全部採用）:
  - M1（「旧と違う点」の、掃引は値が変わらない、という誤り。掃引は既定を使う）: 設計の文を直した。
  - m2（要求1.1の誤記）、m3（要求1.4・4.1の確かめ方）、m4（実旧と比べるtestの期待値を変えない）、m5（版5だけのケースを足すときに広げる3か所）、m6（現在の版を5へ替える時点）: 要求・設計・taskへ書いた。
- 仕様レビュー3回目（修正の再確認。同じ担当・effort、session `7efa6625-3f70-4e37-8b8f-de39235fe4d8`）: 2回目のMajorは解消。新しいMajor 1で`CHANGES REQUIRED`。採否（採用）:
  - M1（ADWINの2構成でも、この場面が数えられうる。版4・5で動作が同じなので、「事象が1以上なら差がある」の検査と矛盾する）: 事象を、次の版で警報を待つ扱いを使うケースだけで数える定義に改めた（ADWINの2構成と、FedDrift・FedAvgは、定義により0）。要求3.2・3.3と設計へ書いた。
- 仕様レビュー4回目（修正の再確認。同じ担当・effort、session `eebab257-248b-4548-ac67-53eafa788bba`）: `SPECIFICATION: PASS`（Blocker・Majorなし。Minor 5）。採否（全部採用）: m1（版5のgoldenを生成する前の中間のcommitで、版5のgoldenの存在のtestが失敗する）・m2（task 1の事象の数え方の範囲）・m3（要求1.4のtestをtask 2の本文へ）・m4（既定に依るtestの候補に、構成の派生のtestのADWINの補助関数を足す）・m5（古い注記）: taskへ書いた。Minorだけなので、再レビューは受けない。
- task 1: 版4の扱いで、版4の全19ケースの事象（`alarm_wait_situation_count`）を数えた（Windows）。`fedsda-sine2-events`が10、`fedsda-sine2-reference-reuse`と`fedsda-sine2-alternative-reference-reuse`が16、ほかの16ケースは0（ADWINの2構成は、定義により数えない）。1以上のケースがあるので、ユーザーへの確認は不要。基準版1〜4の照合は、値を変えずに通過（goldenのtest 165 passed）。
- task 2: 構成の既定のtest（新規`test_fedsda_default_waited_registration_and_alarm_wait.py`）で、REDを確かめた（既定の登録の扱いが違う）。既定を替えた後の全pytestで失敗したのは、既定に依るtestの12件だけだった（依存境界2件——最終構成のmoduleのimportの入替えと、構成のmoduleの`RAISE_ALARM_IMMEDIATELY_POLICY_NAME`の追加——、掃引のrunの識別子2件——`72f2b5b47a95`→`ecef3c8ebd73`——、最終構成の既定の照合6件、構成の派生の構造の比較——test側のADWINの2構成の定義に、即時の警報を明示した。実旧と比べる設定は、もともと即時なので、照合の値は変わらない——、共用のfresh scriptの方針の名前）。実旧と比べるtestの期待値は変えていない。合わせた後、対象のtest（構成の派生の実旧との照合、基準版1〜4の照合を含む）3452 passed。前のspecの、全部登録を既定とするtestは、このspecのtestファイルへ置き換えた。
- task 3（版5の定義）: 版5の全19ケースを、生成の前に、版4と比べて実行した（Windows）。事象が0の16ケースは、版4と同じ値（ADWINの2構成を含む）。事象のある3ケース（sine2）は、差がある。設計からの変更: 版5で、`fedsda-sine2-alternative-reference-reuse`が第2区間の差の不足の判定を通らなくなり、代わりに`fedsda-sine2-reference-reuse`が、別の保有モデルの再利用と第2区間の差の不足の両方を、再び通った。必須の経路は、版5でも全部どれかのケースが受け持つので、seedのケースを足さずに、版5の必須の経路の割り当てを合わせた（設計に書いた）。WSLの生成で、`fedsda-sine2-alternative-reference-reuse`が、版5で再利用の経路も通らなかった（Windowsでは通った。数値の環境の違い）。両方の経路を`fedsda-sine2-reference-reuse`が両方の環境で通るので、`alternative`のケースには、版5で経路の必須を課さない形に改め、sourceをcommitし直して、両方の環境で生成し直した。
- 基準版5の生成（source `2995689`、commit `5d3a425`）: Windowsの基準環境とWSLで生成し、19ケースを`behavior_regression_goldens/v5/`へ置いた後に、現在の版を5へ替えた。照合: Windows 204 passed（goldenのtestの全体）、WSL 219 passed（goldenのtest、このspecのtest、待つ登録のspecのtest）。研究室サーバ用は、対象外。
- task 4（文書）: 実験の入口の文書、基準の文書（版4の明示、版5）、IMPROVE-006の記録と一覧（採用、既定）、LEGACY-016の記録、再開案内、比較scriptの説明の文を更新した。
