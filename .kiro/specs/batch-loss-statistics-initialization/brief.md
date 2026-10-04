# 目的と範囲

前specの統計storeへ渡す新規学習モデルのbatch初期統計を移植する。外部計算済み損失と正解ラベルを受け、全体/classの集計だけを返す。
Boundary Candidates: 数値seed作成。モデルのprepare/forward、学習、登録/送信、初期事前学習の逐次Welfordは範囲外。
