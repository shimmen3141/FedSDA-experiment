# 要求: 保有モデルの学習バッチ抽出

## 目的と範囲
一回共同更新に先立つ参加選別と復元なし抽出を移植する。入力はモデルID順序付きの観測標本列、保有モデルID集合、バッチ標本数、外側が所有する乱数生成器。
出力は参加順のモデルIDと連結された特徴/ラベル。FIFO所有・帰属判断・標本の追加/統合・分類器生成/optimizer管理・学習/反復・診断counter・新全体runは含めない。
旧748c3aaを固定参照とし、旧production/goldenを変更しない。新入力は特徴/観測ラベルだけを持ち、真の概念IDを必要としない。

## 1. 選別と復元なし抽出
1.1 When 正当な入力で抽出を呼ぶ, the Training Batch Sampler shall モデル別標本列の入力順を保持し、保有集合に属して十分な標本数を持つモデルだけを参加させる。
1.2 When モデルが参加する, the Training Batch Sampler shall 指定数の標本位置を一回の復元なし抽出で選び、抽出順に特徴と対応ラベルを連結して返す。同値標本や同一参照の別位置は許容する。
1.3 While モデルが未保有または標本数不足である, the Training Batch Sampler shall そのモデルの標本内容を検査せず、乱数を消費せずに省略する。
1.4 When 参加モデルが存在しない, the Training Batch Sampler shall 空の結果を返し、乱数状態を維持する。保有集合にしかないIDから列を作らない。

## 2. 事前検査と借用状態
2.1 If 入力構造・ID・生成器の型が契約外、モデル別列のIDが重複、または指定標本数が正整数でない, the Training Batch Sampler shall 最初の抽出より前に理由の分かるエラーで拒否する。
2.2 If 参加予定列の特徴/ラベルが有限の対応する一標本でない、または列内の特徴次元が揃わない, the Training Batch Sampler shall 最初の抽出より前に拒否し、後段の不正でも借用乱数と入力値を維持する。
2.3 The Training Batch Sampler shall 参加予定列の全標本を事前検査する。保有列の標本コレクション自体は構造を検査し、不足列の標本内容は検査しない。未保有列の内容は参照しない。
2.4 The Training Batch Sampler shall 入力列・標本参照・値・gradを並べ替え/削除/変更せず、出力の連結テンソルに独立した格納領域を持たせる。
2.5 The Training Batch Sampler shall 指定された借用乱数の状態だけを抽出に応じて進め、外側のPython/NumPy/Torch乱数、既定dtype/device、勾配有効状態を変更しない。

## 3. 移植と接続の証拠
3.1 The Training Batch Sampler shall 小/大の標本列、非昇順/負ID、未保有/不足/空列が混在する条件で、実旧抽出の参加ID・全特徴/ラベル・抽出順と終端乱数状態に完全一致する。
3.2 The Training Batch Sampler shall FIFOが解放した観測位置を上位で標本へ解決して抽出に渡し、出力IDを分類器/個別optimizerに対応付けて既存共同更新へ渡す明示test-only接続で、学習の値と状態を実旧へ照合できる。
3.3 The Training Batch Sampler shall 旧importや上位実行/保存/帰属/FIFO/モデルへの依存を持たず、機能test・依存境界・fresh process・旧11/最終3goldenを含む全回帰で移植を確認する。
