# 終端の未完了候補検証の確定 — 要求 revision4

研究者が旧結果を維持して構造整理を進めるため、実験終端で件数不足の候補検証を棄却し、保留標本を現行モデルへ回収して再分析用の情報を返す。通常評価に必要な件数へ到達した検証は対象外。

返却後のsession解除・記録一覧への追加は呼出側の責任。正常成功時の旧挙動との一致を対象とし、旧の失敗時の途中記録残存は維持対象にしない。並行更新・非公開改変・OOM・計算overflowは保証外。通常の候補採否、通知・episode進行、検出、新client/全体run、旧実装修正は含まない。

対象は、開始APIで正常に生成され、公開観測APIで収集した検証件数が要求件数未満のsession。要求件数以上は拒否し、検証平均の内容によって件数不足を通常評価へ扱い直さない。回収先と返却IDは、開始時点ではなく確定呼出時点の現行学習帰属IDである。

不足判定と記録件数は、そのsessionで実際に収集済みの検証観測件数という同一の値を用い、0以上要求件数未満を対象にする。外部から渡した件数や学習区間の件数で代用しない。保留標本はCPU float32 stridedの有限Tensorで、特徴shape[1,F]（現行分類器の特徴数F）とラベルshape[1,1]、ラベルは0以上クラス数未満の整数値という既存吸収契約を満たす。標本recordと概念ID列はexact tuple、recordは既存の観測標本型、概念はNoneまたはbool以外のbuiltin intで標本列と同じ長さとする。

## 1. 終端の回収

- 1.1 When 検証中のsessionがないとき、the Incomplete Validation Finalization shall 他入力にアクセスせず、検査・状態変更・乱数消費もせず、確定情報のない結果を返す。
- 1.2 When 検証件数が不足したsessionを実験終端で確定するとき、the Incomplete Validation Finalization shall 保留標本をその時点の現行モデルへ回収し、帰属先・候補・固定参照・学習状態・検証損失列を変更しない。
- 1.3 When 未完了の確定位置を記録するとき、the Incomplete Validation Finalization shall 提案位置と終端までに処理した最後の標本位置のうち大きい位置を用いる。

## 2. 記録と引継ぎ

- 2.1 When 回収を完了したとき、the Incomplete Validation Finalization shall 提案/確定位置、検出器、候補の初期学習区間件数、実際の検証観測件数と、件数不足による棄却であることを不変情報として返し、比較参照と有効な検証平均がないことを明確にする。
- 2.2 When 完了情報を返すとき、the Incomplete Validation Finalization shall 確定呼出時点の現行帰属ID・推定変化点・episode IDを返して呼出側がsessionを解除し適応記録を作れるようにし、帰属変更・切替位置・通知・episode操作を行わない。回収先・返却ID・旧eventの変更前後IDはこの同じIDである。

## 3. 拒否と既存結果の維持

- 3.1 If 終端の入力が不正、または検証件数が要求件数以上のsessionが与えられたとき、the Incomplete Validation Finalization shall 損失収集・標本・計数・統計・帰属・モデル・乱数を変更せず例外で拒否する。不正はsession/帰属/状態ownerの型違い、処理件数がboolまたは整数以外または負、保留概念ID列の型/要素型/長さ違い、保留標本の型/shape/値違い、現行帰属IDの保有状態欠落を含む。型違いはTypeError、値違いと要求件数以上はValueError、保有ID欠落はKeyErrorで区別する。
- 3.2 When 正常な未完了sessionの回収を行うとき、the Incomplete Validation Finalization shall 同一入力・同一基準環境で実旧と標本列/概念計数/損失統計/帰属/モデルparameter/optimizer/乱数状態を同じ値にし、観測0件から要求件数未満、空保留、任意metadataなし、開始後の現行ID変更を扱えるようにする。返却から導出する旧判定の提案/確定位置・検出器・学習区間件数・検証件数・棄却・件数不足理由・比較参照なし・比較平均なしと、旧適応eventの確定位置/検出器/棄却/変更前後の同じ現行ID/推定変化点/episode IDを照合する。欠損平均は新の欠損表現と旧NaNを対応付ける。整数/列挙/位置は完全一致、数値も既承認の実旧対照と同じ基準で照合し、環境差を理由にgoldenを更新しない。
- 3.3 When 本specを完了するとき、the Incomplete Validation Finalization shall 終端回収・拒否時の状態保持・既存の開発用回帰基準を維持したことを確認できる証拠を残す。
