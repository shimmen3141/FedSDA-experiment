# レビューと採否

## 要求
Luna初回NEEDS_FIXESの2指摘を採用した。2.1へbool除外を明記し、3.2から開発用ツール/golden実行手段を除いて独立呼出しと依存境界の観測要件へ限定した。
具体的な検証手段はdesign/tasksへ置く。反復順・empty/0・optimizer継続・非rollbackの契約は整合との確認あり。修正後の独立要求レビュー待ち。

修正版のLuna要求レビューPASSを採用。全12条件を承認し、spec.jsonへ要求のLF正規化hashを記録した。

## 設計・命名
Lunaはnaming revision1をPASS、designをNEEDS_FIXESとした。exact builtin intにより派生整数型も拒否する設計が要求に未記載との指摘を採用し、2.1へ組み込み整数・派生型拒否を明記した。
要求の承認を解除して更新hashで要求・設計を再レビューする。正常入力・数値・反復境界は変更しない。

修正要求・設計・命名（テスト名補完を含むrevision1）のLuna最終PASSを採用。3段階の最新LF hashと命名revision承認をspec.jsonへ保存した。

## Task graph / tasks
同Lunaによる独立sanity判定PASSを採用。全12条件の割当、3責務/順序依存、既存上流完成、task2のtest-onlyと非parallelを確認した。
Task1の組合せが広めでも小さいCPU実旧oracleで一責務を検証できるとの評価を採用。新環境や隠れたsetupは不要。
fresh thread枠は利用できないため、既存の実GPT-6 Luna threadを別レビュー依頼で再利用した（independent_reused_thread）。

## Task 1–2命名補完
Luna revision2 PASSを採用。sample_parameters_beforeはNN parameterとも読めるという有用な提案を採用し、既存sample_values_beforeへ統一した。その他の追加test-local名は明瞭との確認あり。最新revision2/hashを承認して実装へ進む。

## Task 1
Luna独立実装レビューAPPROVED、指摘なし。missingmodule collection errorのRED後、新2moduleで72条件GREEN（主担当5.78秒、独立4.83秒）。実旧の複数反復・逆順binding・全抽出/loss/parameter/grad/optimizer/RNGのexact比較を確認。Ruff check/format成功、Pyright 0 errors。Task2の拒否/空/環境保持とTask3の境界/全回帰は未完了。

## Task 2
Luna初回REJECTEDの有用な指摘を採用。testのset_default_device復元は内部DeviceContextを残すため、with torch.deviceの一時scopeへ変更した。scope後device値も確認し、再レビューAPPROVED（独立94 passed/5.33秒、主担当94 passed/4.89秒）。production変更なしのtest-only。回数/ID事前拒否、0未参照、未参加payload、NaNのpredraw拒否、借用/frozen、外側RNG/dtype/device/gradmodeと標本保持、空共有、後続拒否時の先行更新/消費RNG保持を確認。
ambient metaの検証は上流specと同じSGDを使用。Torch自身の未初期化Adam step scalar生成はambient deviceに依存するため、metaの初回Adamまで環境非依存とは主張しない。CPUでのAdam/AMSGradの複数回状態一致はTask1/空共有testで確認済み。

## Task 3 / 最終統合
Luna Task3 APPROVEDとfeature GOを採用。指摘なし。独立対象+AST533 passed/7.97秒、主担当全3721 passed/3 skipped/1既存warning/178.13秒。主担当・Luna fresh CPU実4回Adam更新と旧非import成功。全12条件・借用所有・反復/ID順・拒否/skip・state継続・非rollback・exact依存境界・旧固定差分・Ruff/型/依存検査を確認した。
Task1–3を完成とし、責務外の回数算出/pending/interval、標本/optimizer所有、候補進行・同期・新全体runは後続へ残す。検証結果・源内容hash・限界はintegration-validation.mdに記録した。
