# レビューと承認

要件を生成し、Lunaレビューへ提出。承認は実際のレビュー結果後に記録する。

## 要件レビュー1: NEEDS_FIXES / 採用
Lunaが有効な入力型/shapeとclass_countの型/下限の明記を提案。拒否条件を一意に検証するため有用と判断し、既存CPUfloat32[N]/[N,1]・同一N、dense strided、bool除外のbuiltinint>=2を入力契約へ追記した。範囲・数値契約への追加指摘なし。

## 要件レビュー2: dtype指摘の根拠を再確認
LunaはMNIST読込みのint64ラベルを根拠に整数Tensor受理を提案。主担当は旧data/streams.py:35–42のtorch.FloatTensor変換を確認した。登録時のbatchと読込み直後のNumPy型は異なるため、現在の固定CPUfloat32範囲を広げる指摘は不採用と判断し、この実経路の根拠をLunaへ戻して再判定を依頼した。

## 要件最終: PASS
Lunaは通常streamから登録へ渡るbyがfloat32であり、loss計算時のlong変換と異なることを確認してdtype指摘を撤回。9条件と入力契約に未解決の欠落なしとしてPASS。主担当も旧経路と照合し委任承認した。

## 設計・命名 revision 1: PASS
Lunaは旧演算順・singleton非対称・class順と欠落、reduce前入力検査、既存immutable型とstore/損失生成の責務分離、public名と旧oracle用語の区別を確認しPASS。指摘なし、主担当も照合し委任承認した。

## task graph: PASS / independent reused thread
Lunaは保存前draftで9条件網羅、1→2→3の依存・境界と観測可能な完了条件を確認しPASS。明示Depends省略の軽微な提案は不採用。新セッションでも前提が判別できるよう明示依存を保持する。主担当も対応表と実行可能性を確認し委任承認した。
