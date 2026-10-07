# 警報時の学習区間の準備 — 要求 revision2

研究者が警報処理を旧挙動を維持して組み立てられるように、帰属未確定の標本を前区間と変化区間へ分け、前区間の処理が済んだ状態と変化区間を得られるようにする。

## 境界

対象は候補検証sessionがない警報経路。各標本の明示位置・観測済み標本・診断用概念IDを結び付けた情報、現在の学習帰属、正の推定変化区間件数を受ける。CPU float32の既存二値/多クラス分類器を対象とする。位置に対応する標本の供給と検出からの件数推定は呼出側の責務。

最小区間件数による中止、変化区間の評価・吸収・候補検証開始、active session中の警報、FIFO消費、検出器reset、適応event・切替位置・再利用計数・通知、session保持、設定登録、計算量診断、新client/全体runは対象外。並行更新、非公開状態の改変、OOM/overflowや事前検査通過後の予期しない実行失敗の全体rollbackは保証外。

## 1. 保留標本の区間分割

- 1.1 When 警報時の区間準備を要求されたとき、the Alarm Training Interval Preparation shall 保留順の末尾の「保留件数と推定変化区間件数の小さい方」件を変化区間、それより前を前区間として、各区間の位置・観測済み標本・診断用概念IDを元の対応と順序で返す。
- 1.2 When 推定変化区間件数が保留件数以上のとき、the Alarm Training Interval Preparation shall 前区間を空、変化区間を全保留標本とする。
- 1.3 When 保留標本が空のとき、the Alarm Training Interval Preparation shall 両区間を空として返し、標本・統計・計数・評価標本・乱数を変更しない。
- 1.4 When 区間準備を完了したとき、the Alarm Training Interval Preparation shall 区間列の構造を呼出側が変更できない記録として返し、観測済み標本の内容を変更せず、保留位置と最後の観測位置を維持する。警報時に平時容量を1件超えた保留列も切り詰めない。

## 2. 前区間の処理と更新順

- 2.1 When 前区間が非空で現在の帰属IDが正規IDであるとき、the Alarm Training Interval Preparation shall 前区間から既存の評価保存規則と同じ件数・抽出順・容量保持規則で評価標本を保存し、その後に前区間全件を現行モデルへ既存の帰属確定標本の吸収と同じ順序・値で反映する。
- 2.2 When 前区間が非空で現在の帰属IDが負の一時IDであるとき、the Alarm Training Interval Preparation shall 評価標本を保存せず抽出乱数も消費せず、前区間全件の吸収を行う。
- 2.3 When 前区間が空のとき、the Alarm Training Interval Preparation shall 評価保存と吸収を行わず、標本・統計・計数・評価標本・乱数を変更しない。
- 2.4 When 前区間を吸収するとき、the Alarm Training Interval Preparation shall 現行モデルの学習標本、全体/クラス別損失統計、割当概念計数を更新し、概念IDなしの標本は概念計数へ加えない。変化区間、他モデルの標本・統計・計数、モデルparameter/grad/optimizer、現在の帰属IDを変更しない。
- 2.5 When 前区間の処理を完了したとき、the Alarm Training Interval Preparation shall 吸収後の履歴統計を後続の変化区間評価に利用できる状態で返す。変化区間が後続の最小件数に満たない場合でも前区間の処理は省略しない。

## 3. 拒否と対照証拠

- 3.1 If 供給された各標本の明示位置の列が保留位置列と一致しない（欠落・重複・順序不一致・件数不一致を含む）とき、the Alarm Training Interval Preparation shall 状態と乱数を変更せず例外で拒否する。
- 3.2 If 推定件数、現在の帰属IDまたは状態所有者の型/値が不正、現行モデルが未保有、前区間の標本または概念IDが既存の吸収契約に反する不正であるとき、the Alarm Training Interval Preparation shall 評価標本保存を含む状態と乱数を変更せず例外で拒否する。変化区間の分類器による評価は後続の区間解決の責務とする。
- 3.3 When 同じ入力・環境・初期乱数状態で本処理を実行するとき、the Alarm Training Interval Preparation shall 実旧の警報処理と区間分割、前区間の保存標本順・吸収結果・帰属ID・Python乱数の最終状態を一致させ、torch/NumPy乱数を変更しない。

不正入力の具体的な型・値の一覧と例外分類、事前検査で既存吸収契約を満たす方法は設計で定める。明示位置とpayloadの意味上の正しい対応は供給側が保証する（同形状のpayloadの入替えを値から推定して検出することは保証しない）。変化区間の標本は位置・record・1標本対応を検査し、分類器に依存する損失評価は後続へ残す。LEGACY-002を含む旧挙動の修正は今回行わない。
