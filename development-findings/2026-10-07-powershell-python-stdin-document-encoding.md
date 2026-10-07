# PowerShellからPythonへの標準入力で日本語文書が文字化けした

## 観測

2026-10-07、candidate-epoch-trainingのTask4レビュー記録と統合検証草稿を、日本語を含むPowerShell here-stringからPythonへpipeして書き込んだ。期待はUTF-8日本語の保存だったが、実際は日本語が「?」になった。Task4記録はcommit e833bedにも含まれ、UTF-8で読み直した際に発見した。対象はreview.mdのTask4追記と未追跡integration-validation.mdの草稿。ソース・テスト・承認正本には影響しなかった。

## 影響と対応

読みづらい検証記録になり、同じ経路で日本語の検索文字列をPythonへ渡すと一致せずValueErrorになった。apply_patchとPowerShellのSystem.IO.File.WriteAllText（UTF8Encoding）による直接書込みで修復し、完成レビュー前にGet-Content -Encoding UTF8で保存結果を確認した。数値検証の実行コードは変更していない。

## 改善先と提案

改善先はagent-runtimeのWindows shell手順。ASDD plugin固有の問題ではない。今回のpipeではPythonへ到達する前にUnicodeが失われたと推測するが、shellの全encoding設定を特定したわけではない。日本語のファイル編集にはapply_patchやUTF-8の直接書込みを使い、Pythonのencoding指定だけでpipeを安全と判断しない。別エージェントも編集直後に実ファイルをUTF-8で読み直す。

## 改善結果

今回の2文書は修復済み。review.mdには発生と対応の記録を保持し、feature最終レビューの対象に含めた。環境全体のencoding設定変更は行っていない。
