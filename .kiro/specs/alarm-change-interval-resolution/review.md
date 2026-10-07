# 独立レビューと採否

主担当Claude Code。要求r1・設計r1・命名r1の草案を独立GPT-6 Lunaへ提出する。未承認のsource/testはない。

## 要求・設計・命名 r1からr2

独立fresh CLI GPT-6 Luna（read-only、session `01a117b4-e645-79a3-a0fc-939d54037f32`、HEAD 76b0550）が3文書ともr1をREJECTED。r1のLF hash: 要求790b4bb6a7c3854c234764109644c24cdcfb29917c45288e67435631662442ef、設計475f058b38313f971562b303eb37ea53d9f83446a85314484c32e83412302a2d、命名5d932f68ba93005490a133755272f311537759fdcdcd4058d077a43a94783da0。全指摘を有用と判断して採用した。

- Major（要求3.3に対照条件の列挙とgolden方針が混在）→ 3.3は「実旧と同じ値にする」だけにし、対照条件とgolden方針をbriefの開発ゲートへ移した。
- Major（要求3.1の不変の保証が広すぎる）→ 保証の対象を「事前検査または既存部品の入力検査で拒否される不正」に限定し、検査通過後の更新途中の失敗を保証外へ明記した。3.2も同じ限定にした。
- Major（設計の拒否時不変の根拠が不足）→ 手順ごとに、拒否が起きる場所・その時点までの処理・状態と乱数が変わらない根拠を表にした。snapshotと初期値選択が新しいTensorを作るだけであること、session開始の検査が候補生成（最初の乱数消費）より前に完了することを明記した。
- Major（吸収と切替の順序の等価性の説明が不足）→ 旧の切替の副作用（現行IDの変更と通知の2つ）、通知が吸収の更新する状態を読まないこと、旧の吸収が現行IDを読まないことを根拠として節を追加した。通知は範囲外で呼出側が行う。
- Major（結果種別の値が確定処理の値と紛らわしい）→ 3値ともalarm_interval_の接頭辞を付けた。
- Minor（追跡表の証拠の粒度）→ 要求ごとにtestで比較する項目を書いた。
- Minor（設計で使う局所名の登録不足）→ 保有状態のsnapshot、現在ID、変更記録とsessionのlocal、先頭標本等を追加した。
- 要求3.2（開始だけの入力は再利用・維持では検査しない）は受け入れ可能、既存公開APIだけで実現可能、連結の値とsession開始への引数対応は旧と一致、との確認。

## 要求・設計・命名 r2からr3、承認

r2の再レビューをcodex exec -m gpt-6-lunaへ依頼したが、Codexの利用上限（2026-10-08 5:08まで）に達して実行されなかった（session `01a117b7-3b57-7d83-888f-dbb7bcaa4710`はレビュー結果を返していない）。CLAUDE.mdの取り決め（Luna利用不能時だけ独立したAgentへSonnetを指定して代替する）に従い、主担当とは別contextの独立Agent（model指定sonnet、agent id `ae48502d8c91f73d4`、ファイル変更なし・test未実行）がr2をレビューした。代替理由はLunaの利用上限だけで、Claude担当であることは理由にしていない。

r2: 要求・設計・命名ともAPPROVED、Blocker/Majorなし、Minor 3件。r1の全指摘の解消、拒否時の状態保持の表と既存コードの一致（吸収の検査と評価が更新より前、session開始と候補生成APIの検査が最初の乱数消費より前、snapshotと初期値選択が入力と乱数を変えない）、吸収と切替の順序の根拠と旧コードの一致、許可依存の必要十分、同名同義とした名前の実在を、行番号つきで確認した。Minorは全件採用した。
- 計算量診断が範囲外に書かれておらず、旧の吸収が診断counterも書くのに「だけ」としていた → 要求・設計・READMEの範囲外へ追記し、記述を直した。
- 事前検査のTensorがexact型かisinstanceか未定で、派生型の標本が分岐によって通ったり拒否されたりする → exact Tensorと明記し、拒否matrixへ追加した。
- LEGACY_ACTIONS_BY_RESOLUTION_OUTCOMEが既存定数と1字違い、RESOLUTION_CASESが既存と紛らわしい、assigned_model_idの型の違いが未記載 → 改名と明記。

r3: 同じAgentが差分を再確認し、3文書ともAPPROVED、指摘なし。LF hash: 要求715c3d8722797f77caa2ba609289f0631ce91bf9f1a646d8f9b570a94c7f8bf8、設計88244ab34884ec6185af8e82e949c7d49dc60a3dc930a451209e25c4aed799c1、命名98b8b3d767a56fd0c29d11a361582a76c869013058153f02ef18172d5a3af6c1。

## tasks r1からr2

Lunaの利用上限が続いているため、別の独立Agent（model指定sonnet、agent id `a3af46e032693d1cd`、ファイル変更なし・test未実行）がtasks r1（LF hash 28c5decb947f6bb81e048ee4f3f87f22f61de110f1790104c5020bf943747119）をレビューし、REJECTED。要求10項目の対応、依存順に循環なし、完了ゲート、境界外の混入なし、設計・命名との矛盾なしは確認済み。全指摘を採用した。

- Major「Task 1（再利用・維持だけ）の時点で、適合なしの入力の扱いが読み取れない。暗黙の例外やNone返却が入るとTask 2で作り直しになる」→ 採用。提案は未実装の明示エラーだったが、未実装の分岐を持つ中間状態をcommitしない構成へ変えた。Task 1は結果種別と結果recordだけ、Task 2で3分岐を1つの公開関数として実装する。未実装用の例外や名前は追加しない。
- Minor「Task 1とTask 2の依存importの分担が曖昧」→ Task 1はrecordの定義に必要な型だけ、Task 2で型注釈用の型と呼び出す部品を追加、と書いた。
- Minor「test局所名の事前登録の扱いが冒頭の一文では弱い」→ naming.mdのTest/Evidence節に登録済みで、不足時はtestを書く前に追加レビュー、と書いた。

tasks r2: 同じ独立Agent（sonnet、`a3af46e032693d1cd`）が再レビューし、APPROVED、指摘なし。r1の3指摘の解消、Task 1がrecordだけのmoduleとして単独でtest・commit可能であること、要求10項目の対応と依存順、Task 2が設計の手順1〜5と対応することを確認した。LF hash=e280b69e4764454826e4e7255535774de4171ea7eff14d1b4dcb74518f613f79。source/testは未着手、testは未実施。

## 命名r4（Task 2/3のtestで使う名前）

Task 2のtestをリポジトリ外で下書きして表にない名前を洗い出し、testをリポジトリへ置く前にnaming.md末尾の「追加 revision4」へ登録した。Lunaの利用上限が続いているため、要求・設計・命名をレビューした独立Agent（sonnet、`ae48502d8c91f73d4`）がレビューし、APPROVED（Blocker/Majorなし）。レビュー時のLF hash=e981236826178f99af872d3e57d30ff4be706dbd27b5b5d8e255ba1aaee4486d。同名同義とした名前とhelperの既存testでの実在、記録用関数の名前、build_fixation_oracle/assert_started_session_matches_legacyを使わない理由（区間を新しく連結するため同一object照合が成り立たない）を確認した。

- Minor「state_snapshotのkeyのうち今回追加の対象にresolution_argumentsとcurrent_training_model_idが漏れている」→ 採用。該当行の文言だけを直した（名前の追加・変更なし）。この文言修正後のLF hash=2f2466c4422ffbb4f274483d1af3a9f24aa7f8b8b7ddb2fbed873a0d666419f2は、レビュー担当が見た内容と該当1行の説明だけが違う。
- Minor（新規の局所名が新規として書かれていることの確認）→ 対応不要。

## Task 1/2の実施記録

Task 1（commit 1f6fd4a）: test先行。実装ファイル作成前に対象testを実行し、collection時のImportErrorでREDを確認。依存境界は注入契約test追加時に21 failed/1793 passedを確認してからguardと両resolver登録を追加。実装後、依存境界＋対象で1815 passed。Ruff/対象Pyright成功。

Task 2: test先行。testはリポジトリ外で下書きして命名r4を登録・承認後にリポジトリへ置き、公開関数の実装前に実行してcollection時のImportError（resolve_alarm_change_intervalなし）でREDを確認。依存境界は公開関数ぶんの注入契約へ更新して59 failed/1957 passedを確認してからguardを更新。実装後の初回実行で依存境界＋対象2197 passed（対象181）。

- 対象Pyrightが1件指摘した。既存の初期値選択の戻り値型は`dict | None`で、session開始は`dict`を要求する。保有モデルが1件以上あれば初期値選択はNoneを返さないが、型上の分岐を明示し、Noneの場合はLookupErrorで拒否する3行を加えた（到達しない分岐。設計の手順5に挙動の変更はない）。対象Pyrightは0 errors。
- 実source変異22種のうち初回は21種を検出。未検出は「標本の1行検査の削除」で、拒否testの2行標本が特徴とラベルを同時に2行にしていたため、ラベルの形状検査が先に拒否していた。特徴だけ2行の条件（sample_two_feature_rows_with_one_label）をtestへ追加し、22/22検出。元byteの復元と復元後の全対象passedを確認。productionは無変更。対象は184、依存境界＋対象で2200 passed。
- 全分岐のtestが初回からGREENだったため、上の変異で検出力を確認した（分岐条件、吸収先、切替の欠落、吸収の欠落、概念ID、標本順、連結順、結果種別、初期値選択へ渡す候補列、保留標本、候補検証開始での吸収、余分な乱数、位置情報、各事前検査、IMPROVE-001相当の現行優先への変更）。
