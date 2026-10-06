# 要求 revision 1

## 1. 固定

- 1.1 When 保有モデルの一覧が渡されたとき, the fixation operation shall 保有している各モデルについて、一覧の順に、同じ構造で保有モデルと同じparameter値を持つ新しい分類器を作り、モデルIDから分類器への対応として返す。
- 1.2 The fixation operation shall 各参照分類器を保有モデルから独立させる。共有特徴抽出部を含め、保有モデル・他の参照分類器とparameterの実体を共有せず、以後の保有モデルの更新が参照分類器の値へ及ばない。
- 1.3 The fixation operation shall 各モデルの全体損失統計の件数が2以上のとき、その保存平均（0を含む）を参照比較用の履歴平均損失として、モデルIDからの対応に含めて返す。件数が2未満または統計が未登録のモデルは対応に含めない。
- 1.4 The fixation operation shall 保有モデルの一覧、分類器のparameter/grad/optimizer、損失統計を変更しない。CPUのtorch乱数は、一覧の順に分類器を1つずつ生成する量だけ消費し、Python/NumPyの乱数は消費しない。

## 2. 呼出し契約

- 2.1 If 一覧または統計storeが対象の具体型でないなら, the fixation operation shall TypeErrorを送出し、乱数を消費しない。
- 2.2 If 保有モデルが一つもないなら, the fixation operation shall LookupErrorを送出し、乱数を消費しない。
- 2.3 If いずれかの保有モデルの分類器が既存のparameter snapshotの検証で拒否されるなら, the fixation operation shall 例外を伝え、どの分類器も生成せず乱数を消費しない。

## 3. 境界と移植

- 3.1 The fixation operation shall 一覧と統計storeを読むだけで、自身の永続状態を持たない。候補の生成と学習、損失収集の開始、参照分類器の学習やoptimizerの生成、検証標本の観測、採否評価、確定を行わない。
- 3.2 When 同じ保有モデルと統計へ固定を適用したとき, the new fixation operation shall 実旧の参照複製による参照IDの順序、各参照の全parameter値と出力、独立性、処理後のtorch乱数状態を維持し、実旧のsession開始が取得する履歴平均と同じ対応を返す。固定した参照と履歴平均で観測・評価・確定を行うと、実旧のsession開始から確定までと同じ状態になるようにする。
