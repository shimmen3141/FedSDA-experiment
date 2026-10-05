# 要求: 一回の共同モデルパラメータ更新

## 目的と範囲

最終Residual Adapter構成の標本数加重平均による共同学習を、明示された参加バッチ列に対する一回の更新として移植する。
参加選別・バッチ抽出・反復・optimizer生成/reset/所有・モデル登録・候補進行・通信・診断/計算カウンタ・PCGrad・新全体runは含めない。
既存のモデル構造、optimizer生成、ローカル学習設定を利用する。数値回帰の基準は旧748c3aaであり、旧productionやgoldenを更新しない。

## 1. 明示された参加列と設定

1.1 The Joint Update Component shall 共同学習と標本数加重平均の正式設定、共有特徴抽出部、任意の共有optimizer、参加バッチの順序付き列、共有部の更新可否を明示入力として受け取る。
1.2 When 正当な入力の参加列が空である, the Joint Update Component shall 更新も勾配初期化も行わず、更新結果なしを返す。
1.3 If 設定・入力の型・更新可否または参加記録が契約外である, the Joint Update Component shall 更新開始前に理由の分かるエラーで拒否する。PCGradを含む非対応の学習/勾配統合方式も事前拒否する。

## 2. バッチと対象パラメータの整合性

2.1 When 参加列が非空である, the Joint Update Component shall 各バッチが正の標本数、共有特徴抽出部の入力次元に適合する有限の入力特徴、同じ標本数の観測ラベルを持ち、全分類器が同一の共有特徴抽出部を参照することを更新開始前に確認する。
2.2 If 二値ラベルが有限の0以上1以下の値でない、または多クラスラベルが有限の0以上分類器のクラス数未満の整数値でない, the Joint Update Component shall 更新開始前に拒否する。
2.3 If 分類器・個別optimizerが重複する、optimizerのパラメータ参照列が対応する共有部または個別部と一致しない、あるいは更新するパラメータが学習可能でない, the Joint Update Component shall 勾配・パラメータ値・optimizer stateを変更する前に拒否する。
2.4 Where 共有特徴抽出部にパラメータが存在しない, the Joint Update Component shall 共有optimizerなしで個別部の学習を実行する。共有パラメータが存在する場合は、更新可否にかかわらず対応する共有optimizerを必要とする。

## 3. 数値と操作順序

3.1 When 正当な非空列を更新する, the Joint Update Component shall 共有optimizerがあればその勾配を初期化し、個別optimizerを参加順に初期化した後、入力順に連結した特徴から共有特徴を一回計算する。
3.2 When 共有特徴を計算した, the Joint Update Component shall 入力順に分割した特徴から各分類器の平均損失を求め、二値は確率の二値交差エントロピー、多クラスはlogitsの交差エントロピーを用いて、損失と標本数の積の入力順の和を総標本数で割る。
3.3 When 共同損失を得た, the Joint Update Component shall backwardを一回実行し、共有更新が有効なら共有optimizerを一回stepした後、個別optimizerを参加順にそれぞれ一回stepし、更新前に計算した共同損失の数値を返す。
3.4 While 共有更新が無効である, the Joint Update Component shall 共有特徴の計算で勾配を記録せず、共有optimizerをstepせず、個別部を学習する。

## 4. 拒否と状態の境界

4.1 If 非空列で勾配計算が外側から無効化されている、または入力契約に違反する, the Joint Update Component shall optimizerの勾配初期化より前に拒否し、既存の値・勾配・optimizer stateを維持する。
4.2 The Joint Update Component shall 入力テンソルの値、モデル構造、optimizer設定、乱数状態、外側の勾配有効状態を変更しない。正当な更新に伴うパラメータ・grad・optimizer stateの変更は許容する。
4.3 The Joint Update Component shall 更新開始後の演算失敗や非有限計算に対する原子的rollbackを保証しない。入力検証は学習中の数値安定性の保証とは区別する。

## 5. 移植の証拠と依存境界

5.1 The Joint Update Component shall 単一/複数参加、不均等標本数、二値/多クラス、共有更新有効/無効、複数回更新で、実旧共同学習の全パラメータ・勾配・optimizer state・操作順を同一基準環境で照合できる。
5.2 The Joint Update Component shall 空共有部の個別学習を独立検証し、この正常拡張を旧経路との直接照合範囲から明示的に区別する。
5.3 The Joint Update Component shall 旧importや上位実行・保存・CLIへの依存を持たず、対象機能テスト・依存境界・fresh process・旧11/最終3goldenを含む回帰検証で移植結果を確認する。
