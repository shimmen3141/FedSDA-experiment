# task graph draft

順次実行、並列実装なし。命名承認前のsrc/test作成なし。
|task|前提|受け入れ証拠|要件|
|---|---|---|---|
|1|要求/設計/命名承認|state/record/入力検証をTDD RED→GREEN。実旧具象FedSDAの一枠・取得・delay1/2/4・ID/置換/clear/no-op列、拒否時不変/借用参照。旧の保留なし状態では有効残回数を0として比較（research.md参照）。対象pytest/品質とLuna実diff|1.1〜1.4,2.1〜2.4,3.1|
|2|task1承認|6条件の実旧registerと新producer/loss/初期統計/store/pendingをtest-only接続。同ID統計更新後に登録時snapshotと現在統計を全field対照。対象pytest/品質/Luna。test-only RED N/A|1.4,3.2|
|3|task2承認|exact依存注入RED→guard GREEN、fresh新CPU/旧非import、全pytest旧11/最終3golden、Ruff/format/Pyright/pip/固定差分/hash/JUnit、Luna独立確認|3.3|

全3task check/承認後に別feature統合GOを取得する。現在地を更新し通常commit/pushする。統計/登録全体/新client/全体runを完成と扱わない。
