# 候補検証sessionの保持と進行 — 要求 revision1

研究者が候補検証の開始から終了までを1つの流れとして追えるよう、進行中の候補検証sessionを1箇所で保持し、警報応答・標本ごとの進行・実験終端の回収を、適応記録と保持の解除へつなぐ。

## 境界

最終構成（固定参照による候補検証）。完了済みの部品（警報応答、候補検証の進行、未完了の終端回収、適応記録）を呼び出してつなぐ。それらの中の処理（観測、採否、吸収、採用、記録の変換規則）は変更しない。

警報応答そのものの実行と完了処理（監視の再開、保留位置の消費）、警報応答の適応記録、学習帰属変更の通知と保存診断、検出episode、判定記録の一覧、標本ごとのclient進行全体、設定登録、新client・全体runは除外する。並行更新は保証しない。上流の部品が更新を始めた後の失敗の巻き戻しは、上流の既存契約のとおり行わない。

## 1. 保持

- 1.1 The Candidate Validation Session Holder shall 進行中の候補検証sessionを高々1つ保持し、空のときだけ新しいsessionを受け入れ、保持中のsessionを外して返せる。保持中に別の（または同じ）sessionの保持を求められたとき、空のときに解除を求められたとき、sessionでないものを渡されたときは、状態を変えずに拒否する。
- 1.2 When 完了した警報応答を保持へ反映するとき, the Held Candidate Validation Progress shall 候補検証を開始した応答ではそのsessionを保持させ、候補検証中の応答では保持を変えず、それ以外の応答（不足、再利用、維持）でも保持を変えない。
- 1.3 If 候補検証中の応答が保持中のsessionと同じものを指していない、または候補検証中でない応答を保持が空でない状態で受けたとき, the Held Candidate Validation Progress shall 保持を変えずに拒否する。

## 2. 進行と終端

- 2.1 When 保持中のsessionへ標本1件の進行を求められたとき, the Held Candidate Validation Progress shall 既存の進行を1回実行し、要求件数に到達して確定した場合は、候補検証の適応記録を1件追加してから保持を解除する。未到達の場合は保持を続け、記録を追加しない。進行の結果と、追加した記録（なければなし）を返す。
- 2.2 When 実験の終端で保持中の未完了sessionの回収を求められたとき, the Held Candidate Validation Progress shall 既存の終端回収を1回実行し、未完了の適応記録を1件追加してから保持を解除し、回収の結果と記録を返す。
- 2.3 While sessionを保持していないとき, the Held Candidate Validation Progress shall 進行と終端回収のどちらも何も更新せず、記録を追加しない。
- 2.4 The Held Candidate Validation Progress shall 通知・検出episodeの操作・警報応答の記録を行わず、保持と適応記録以外の状態を既存の進行・終端回収を通してだけ更新する。

## 3. 検査

- 3.1 If 保持または適応記録のownerの型が不正であるとき, the Held Candidate Validation Progress shall 既存の進行・終端回収を呼ぶ前（どの状態も更新する前）に拒否する。
- 3.2 If 保持へ反映する警報応答の型が不正である、または結果種別とsessionの組が応答の規則に合わないとき, the Held Candidate Validation Progress shall 保持の更新前に拒否する。
- 3.3 3.1・3.2の範囲は、本処理が直接受け取って使う値（保持、適応記録のowner、警報応答）とする。進行・終端回収へ渡すだけの引数は既存の部品が検査する。既存の部品が返す完了情報は、既存の記録処理が追加の前に検査する。

## 4. 対照

- 4.1 When 同じ正常入力・固定環境で実行したとき, the Held Candidate Validation Progress shall 実旧clientのsessionの保持（警報応答5種類の後、到達時の確定の後、未到達の観測の後、終端回収の後）と一致させ、追加する適応記録を実旧の適応イベントと一致させる。記録の追加は保持の解除より前に行う（旧の順）。
- 4.2 The change shall 固定旧実装とgoldenを変更せず、既存の全回帰を維持する。
