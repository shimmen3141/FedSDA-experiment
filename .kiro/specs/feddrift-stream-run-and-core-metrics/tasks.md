# Implementation Plan

- [x] 1. 設定から全体観測・同期・終端を接続する
  - actual旧全体run oracleを起動確認し、入口stubでREDにする。
  - run条件の共有preflightとFedDrift実行順を実装し、共通結果/イベント/causeへ接続する。
  - 時刻準備とR回client学習/server roundの順、全状態・生成列・RNG・未処理末尾/0 round/再実行、入力/段階失敗の対照で完了。
  - _Boundary: FedDrift全体run実行接続_
  - _Requirements: 1.1,1.2,1.3,2.1,2.2,4.1_
- [x] 2. 全体runから共通の基本指標を導出する
  - 共通計算へ記録を渡し、新FedSDAと同じ名前/単位の基本指標を返す。
  - actual旧基本指標一致、末尾の見逃し、読取り性と不整合拒否で完了。
  - _Boundary: 終了済みFedDrift記録からの指標導出_
  - _Depends: 1_
  - _Requirements: 3.1,3.2,4.1_
- [x] 3. Linux基準と独立起動へ接続する
  - 旧2process再現性を確認した別Linux goldenへ新run/基本指標を照合する。
  - 共用freshへ全体run→基本指標を接続し、exact依存/禁止例と対象検証を通す。
  - _Boundary: 全体runの環境基準と新実装独立起動_
  - _Depends: 1,2_
  - _Requirements: 4.2,4.3_

## Implementation Notes

- task graph gate inline PASS。全体run/読取り指標/環境接続の3責務を依存順に実行する。仕様書群の独立レビュー後にmanual --review offでtask順に実装する。全task後の実装レビューと指摘反映後、最終全pytest/identityで完了判定。Minor以下は修正/採否のみ、Major以上は修正範囲を再確認する。

- 仕様レビュー初回: Luna medium /root/feddrift_whole_run_spec_review、Major 2件。同期内部の旧順を明記し、成功イベントのstage/終端成功境界/部分失敗時の非返却を具体化。実装責務は既存synchronize_timestepの内側loopを再利用する全体接続で維持。修正範囲の再レビュー待ち。

- 仕様レビュー修正範囲PASS。追加指摘はtrain_local_roundのk_steps=detection_batch_sample_countを根拠に不採用（reviewer読み違いと確認）。manual task 1 --review offで実装へ進む。

- task 1/2: stubのNotImplementedErrorで各REDを確認後、対象70 passed。実旧全体run対照（全状態/生成列/local 3 RNG）、共通指標/読取り/不整合拒否、入力preflight/全stage cause・位置・成功境界を確認。

- task 3: Windows対象/依存/fresh 3159 passed・Linux専用3 skipped。Linux同process全状態/基本指標と別goldenは73 passed（46.42秒）。新source単独freshで全体run→基本指標成功。型検査0 errors、Ruff/format322、pip check/names/diff --check成功。続いて不整合拒否4ケースを追加し、全task後のHaiku medium実装レビューを起動（未承認）。

- Haiku実装レビュー初回FAIL、M1採用: 非0切替のLinux goldenを別2process再現で追加。m1不採用: chunk対応はclient設定の値を再検査する必要があり、同runtime内のexact client private設定参照を維持（公開getter追加だけの仕様変更をしない）。m2採用: 途中client/round失敗の位置と先行成功イベントを追試。m3採用: factory契約外入力の検査順変更をdesignへ明記。m4不採用: namesは--base 27d41b0で新公開名のみ照合済み、既存class名の再登録不要。任意のfresh数値強化は不採用、数値一致は対照/goldenが担当しsmokeは接続を検査する。

- M1修正検証: 非0切替の旧Linux別2process完全一致、新Linux golden全4case passed（23.61秒）。m2途中位置の2test passed。追加transition拒否4ケースを含むLinux指標34 passed（31.25秒）。Major修正箇所のHaiku再確認待ち。

- Haiku修正範囲PASS（session 98319505-8aa0-4354-8ff7-860708840f7d、actual claude-haiku-5-5、effort medium指定、is_error=false）。旧drift JSONのsine2を追加goldenのsine2_driftへ対応付け、17基本指標/最終hashの一致を独立確認。中間fail2test/Ruff/diff-checkも成功。独立WSL/全pytest/identityは未実施。主担当は最終全pytestへ進む。

- 最終確認で生成器構築（MNISTファイル読込み）の例外が初期準備wrapper外と判明。初回全pytestは停止（未完走・検証成功に含めない）。失敗注入2ケースのREDを実測し、生成器構築とfactory準備を同じ初期準備wrapperへ接続。成功順/イベントは維持し、exact型注釈2依存のみ追加。変更範囲再レビューと最終全pytestやり直し待ち。

- 初期準備修正後、Windows run44+依存3088=3132 passed、Linux全体/指標/別golden82 passed（42.65秒）。成功時全状態/生成列/3RNGと読取り指標、構築失敗cause/段階/以前のprepared保持を確認。

- 初期準備変更範囲のHaikuレビュー: ロジック/3132pytest/fresh16/Ruff成功、境界testのCRLF formatのみFAIL（session128b8c79-bcdc-4186-b8ab-17b5da27fd6b）。LF正規化後format322/diff-check独立PASS（session65fd02e4-7e66-4d24-b07d-4dd6ad8946f6）。actual claude-haiku-5-5/medium指定/is_error=falseを確認。前回と合わせ実装承認PASS、最終全pytestを再実行。

- 最終全pytest: 13387 passed・8 skipped・8 warnings、exit0、673.53秒。JUnit13395/失敗0/エラー0、旧11・最終3goldenを含む回帰2testcase成功。前spec13307+今回88（run44/metric34/Linux4/境界6）=13395。新Linux4skipは環境専用、従来4skipは維持。最終Ruff/format322、Pyright0、pip check、names、progress、diff-check、固定旧差分なし。検証対象commit/source identityはsource commit後に記録する。

- 完了判定GO。検証対象source commit `1f58084dabd8426bd336803b4c4d35719b482df7`、LF source SHA-256 `d4a4c41798c777876d292ddac38cd43fffd749a19cba081b5504160c61bc977c`（426 paths）。固定旧差分/clean tree/JUnit/回帰/件数のidentityを確認し、完了文書commitでも同hashを再照合する。独立全pytest/WSLは未再現、主担当実測と区別した。計算量/変更応答・モデル・予測診断/成果物/CLIは後続。
