# 保有モデルの共同学習反復

## 目的と範囲
研究実装の保守者が、新実装の一回バッチ抽出と一回共同更新を組み合わせ、
外側が指定した回数だけ反復できるようにする。現在はこの接続がテスト側だけにある。
旧の共有部・個別部の更新順、標本抽出順、optimizer継続状態を維持する。
最終構成の標本数加重平均勾配を対象とし、共有更新あり・共有凍結を扱う。
回数の算出・更新間隔・保留更新数・観測標本の所有/帰属・optimizer生成/reset・
候補学習・通信・counter/時間・勾配診断・PCGrad・新全体runは含めない。
旧基準748c3aaとgoldenを変更しない。

## 1. 指定回数の共同学習
1.1 When 正当な入力と非負の反復回数を渡す, the Joint Training Iteration Executor shall 指定回数だけ、各回で新しくバッチを抽出してから共同更新へ進む。
1.2 When バッチが抽出される, the Joint Training Iteration Executor shall モデル別標本列の順序でモデルIDを分類器と個別optimizerへ対応付け、共有部と参加個別部を一回共同更新する。保有モデルの対応表の順序で参加順を変えない。
1.3 While 参加可能なモデルがない, the Joint Training Iteration Executor shall その回の更新を省略し、モデル・optimizer・乱数状態を変更しない。
1.4 When 反復回数が0である, the Joint Training Iteration Executor shall 学習入力へアクセスせず、抽出・更新・乱数消費を行わない。
1.5 The Joint Training Iteration Executor shall 各実行済み共同更新の更新前損失を実行順で返す。参加のない回に架空の損失を追加しない。

## 2. 借用状態とエラー
2.1 If 反復回数が真偽値を除く非負のPython組み込み整数でない, the Joint Training Iteration Executor shall 乱数消費とモデル変更の前に拒否する。整数の派生型も受理しない。
2.2 If 正の回数で保有モデルの対応表の構造・IDが不正またはIDが重複している, the Joint Training Iteration Executor shall 最初の抽出より前に拒否する。
2.3 The Joint Training Iteration Executor shall 指定された乱数生成器とモデル/optimizerを借用し、回ごとにコピー・再生成・初期化・乱数の巻戻しを行わない。
2.4 The Joint Training Iteration Executor shall 標本列・帰属・モデルID・設定を変更しない。抽出後の学習入力検査や演算で失敗した場合、消費した乱数や以前の完了更新を巻き戻す保証をしない。
2.5 The Joint Training Iteration Executor shall バッチ抽出と一回共同更新が受理する正常入力・共有凍結・空共有部を維持し、外側の乱数・dtype/device・勾配有効状態を変更しない。

## 3. 移植と検証
3.1 The Joint Training Iteration Executor shall 実旧の共同学習反復に対して、二値/多クラス、複数optimizer、複数反復、保有外/不足列、参加順で、全パラメータ・grad・optimizer状態・終端乱数を照合できる。
3.2 The Joint Training Iteration Executor shall 旧実装・上位の実験実行・FIFO帰属・保存へ依存せず、独立した呼出しで学習できる。
