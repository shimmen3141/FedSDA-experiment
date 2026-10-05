# 実装タスク

- [x] 1. モデル別標本保持を実旧追加・ID対応へ照合する
  - 新module未存在の実RED後、標本の追加・順序付き読み出し・一回ID対応を実装する。
  - 実旧absorb/emptyextend/apply_server_mappingと全標本参照・順序を比較し、衝突/負ID/空列/連鎖/循環/重複を確認する。
  - snapshot後の追加・対応で旧snapshot構造が保持されること、payload借用と後尾不正入力でstate不変を検証する。
  - 完成は対象test GREENとLuna実装APPROVEDで確認する。
  - _Boundary: 保持器と実旧対照test_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3_

- [x] 2. 保持snapshotから実学習batch抽出へ接続する
  - test-onlyで実旧に独立に保持した標本と新snapshotから抽出し、全モデル順・batch Tensor・終端Randomを比較する。
  - ID対応の有無/衝突/循環、空列、未保有、不足標本、複数batch件数/seed/反復を組み合わせる。
  - Tensor異常は保持時に触らず参加samplerで拒否することを確認する。
  - 完成は母集団/抽出の実旧一致、既存sampler production無変更、Luna実装APPROVEDで確認する。
  - _Boundary: 上位外側test-only接続_
  - _Depends: 1_
  - _Requirements: 1.4, 1.5, 2.1, 2.2, 3.1, 3.2_

- [x] 3. 依存境界・全回帰・完成証拠を揃える
  - exact禁止/許可import注入を先行し実RED後にAST guardを追加する。
  - fresh新package CPU smoke、対象/全回帰（旧11/最終3golden）、Ruff/format/Pyright/pip、旧固定差分/源内容hashを確認する。
  - 全10要件と実測/限界/旧所見を記録し、task状態/roadmap同期後にLuna最終feature GOを確認する。
  - 完成はTask APPROVEDとfeature GO、現在の証拠hash一致で確認する。
  - _Boundary: 依存test/対象spec/roadmap_
  - _Depends: 1, 2_
  - _Requirements: 3.1, 3.2_

## Implementation Notes
番号付きmanual実装、独立Lunaレビューrequiredで順に実行する。
旧oracleの診断/statisticsは外側で無作用化するが、実旧append/remap/samplerを置換しない。
旧samplerへglobal Random開始状態を借用Randomに合わせ、finally復元する。
snapshotはTensorのdeep copyではない。正式登録pop/上書きは今回のremapへ統合しない。
日本語ファイルはapply_patchで書き、PowerShell経由Python stdinへ日本語literalを送らない。
