# 要件: モデル別評価標本の保持

既にモデルへ帰属した評価標本の列構造を所有する。payload・採番・評価forward・評価対象fallback・学習・正式登録全体・通信は外側責務。呼出しで借用する独立Randomを用い、共有乱数は使わない。

## Requirement 1: 抽出追加と容量

1. The Evaluation Sample Store shall 正のbuiltin intのモデル別最大保持件数と、非負builtin intの追加時抽出件数を固定条件として受け取る。bool/派生型を拒否する。
2. When 検証済み負IDへの追加を要求される, the Evaluation Sample Store shall 状態とRandomを変えない。
3. When 非負IDへ追加する, the Evaluation Sample Store shall 初出順に空列を登録し、入力件数と抽出件数の最小件数を復元なしに抽出して追加する。0件ではRandomを消費せず、全件でも抽出順を使う。
4. When 追加後に最大保持件数を超える, the Evaluation Sample Store shall 末尾の規定件数を保持する。標本順・重複参照を保ちpayloadを変更しない。

## Requirement 2: ID対応

1. When 登録済み元IDを単一付替えする, the Evaluation Sample Store shall 元列をpopし先へ上書きする。先既存の位置を維持し、新先/同IDは末尾とする。元欠落はno-op、容量再抽出とRandom消費はない。
2. When サーバ対応表で再編する, the Evaluation Sample Store shall 元モデル順にIDを一回だけ対応し先初出順に列を連結する。空列・未対応IDを保持し、連鎖を再帰適用しない。
3. When 再編した列が容量超過する, the Evaluation Sample Store shall 先初出順に超過列だけ規定件数を復元なし抽出して保持する。非超過列は抽出せず順序を維持する。

## Requirement 3: 所有・拒否・接続

1. The Evaluation Sample Store shall 追加のexact tuple/各exact評価標本record、全IDのexact builtin int、対応表のexact dict/全key値、追加/再編のexact Random引数を状態/RNG変更前に検証し、不正項目と理由を示して拒否する。負ID/0件/空列/元欠落でも該当入力を検査し、追加/再編では実際の抽出がなくてもRandom引数を検査する。
2. The Evaluation Sample Store shall 一覧snapshotを不変record/tupleで構造分離して返し、取得済みID/列を後続操作で変えない。record/Tensorは借用し、opaque/数値型/device等payloadを検査・copyしない。
3. The Evaluation Sample Store shall 評価標本以外のモデル・学習標本・統計・counter・送信保留・共有乱数と数値環境を変更せず、旧実装・runtime・CLI・保存・設定globalへ依存しない。
4. The Evaluation Sample Store shall 旧正常条件の追加→単一付替え→対応表再編と、取得した標本を使う既存新分類器損失評価への明示接続で、列/参照/Randomと損失を一致させる。
