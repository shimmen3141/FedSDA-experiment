# 要求 revision 2

## 1. 採用の組立

- 1.1 When 採用された候補が渡されたとき, the adoption operation shall 採番ownerの次の一時IDを新モデルのIDとし、そのIDで初期ローカル登録（共有反映、学習状態一覧、初期損失統計、送信保留と待機）を行い、採番ownerの次の値を1つ進める。
- 1.2 When 登録を終えたとき, the adoption operation shall 候補の学習標本数とparameter更新回数を同じ一時IDの学習計数へ加算し、渡された保留標本を同じ一時IDの学習標本として渡された順に追加する。保留標本が空でも一時IDの標本列を作る。
- 1.3 When 計数と標本を更新したとき, the adoption operation shall 最後に現在の学習帰属IDを一時IDへ切り替え、変更前後のIDの記録を返す。通知・ログは発行しない。
- 1.4 The adoption operation shall 既存モデルの損失統計と割当概念計数、既存モデルの学習標本・学習計数、評価標本を変更せず、追加した標本の損失評価を行わず、乱数を消費しない。

## 2. 呼出し契約

- 2.1 If 採番owner・学習標本store・計数storeが対象の具体型でないなら, the adoption operation shall TypeErrorを送出し、どの状態も変更しない。
- 2.2 If 候補の学習標本数またはparameter更新回数がbuiltin int以外または負なら, the adoption operation shall TypeErrorまたはValueErrorを送出し、どの状態も変更しない。If 保留標本がexact tupleでない、または要素が学習標本recordでないなら, the adoption operation shall TypeErrorを送出し、どの状態も変更しない。
- 2.3 If 次の一時IDが学習標本store・計数storeで使用済み、または現在の学習帰属IDと同じなら, the adoption operation shall ValueErrorを送出し、どの状態も変更しない。
- 2.4 If 初期ローカル登録が入力を拒否するなら, the adoption operation shall その例外を伝え、採番ownerを含むどの状態も変更しない。登録の拒否条件には、次の一時IDが学習状態一覧・損失統計store・既存の送信保留の対応IDで使用済みであること、保有モデルが一つもないこと、待機ラウンド数・登録先owner・候補・個別optimizer管理器・特徴・ラベルの不正が含まれる（初期ローカル登録の要求2.1〜2.6）。
- 2.5 The adoption operation shall 2.1〜2.3の検証を終えてから初期ローカル登録を呼び、登録が自身の全検証と数値生成を終えて状態更新を完了した後にだけ、採番の確定・計数・標本・現在IDを更新する。検証済み入力に対する既存owner APIの順次更新であり、private状態の改変・並行更新・メモリ不足等による途中例外へのrollbackは提供しない（初期ローカル登録の要求2.7と同じ限定）。

## 3. 境界と移植

- 3.1 The adoption operation shall 各ownerを借用して既存APIで更新し、自身の永続状態を持たない。採否判定、判定記録、切替位置・検出エピソード・適応イベントの記録、現在ID変更の通知、候補sessionの破棄、計算量診断、送受信を行わない。
- 3.2 When 有効な同じ状態へ採用を適用したとき, the new adoption operation shall 実旧の前向き検証確定処理の採用分岐による一時ID、保有一覧、共有部と全モデルの値、損失統計、送信保留と待機、学習計数、学習標本の列と順序、現在の学習帰属IDを維持する。採用後の共同学習が実旧と同じ数値で継続できるようにする。
