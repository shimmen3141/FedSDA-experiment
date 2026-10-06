# 要求 revision 1

## 1. 採番

- 1.1 When クライアントIDを渡して採番ownerを生成したとき, the allocator shall 次に採番する一時IDを -100 − クライアントID とする。
- 1.2 When 採番を要求されたとき, the allocator shall 次に採番する一時IDを返し、次の値を1だけ減らす。返す値は常に負で、同じownerが同じ値を二度返さない。
- 1.3 The allocator shall 次に採番する一時IDを、採番せずに読み取れるようにする。

## 2. 呼出し契約

- 2.1 If クライアントIDがbuiltin int以外なら, the allocator shall TypeErrorを送出する。If クライアントIDが負なら, the allocator shall ValueErrorを送出する。bool/int派生型/暗黙変換を受理しない。

## 3. 境界と移植

- 3.1 The allocator shall 次の一時IDだけを状態として持ち、乱数・数値ライブラリ・設定・他のownerへ依存しない。採番したIDでのモデル登録、現在の学習帰属IDの変更、正式IDへの付替え、クライアント間の一意性の保証を行わない。
- 3.2 When 同じクライアントIDで同じ回数だけ採番したとき, the new allocator shall 実旧クライアントの初期値と採番処理が返すID列と同じ列を返す。
