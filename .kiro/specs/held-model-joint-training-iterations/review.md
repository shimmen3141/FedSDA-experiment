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
