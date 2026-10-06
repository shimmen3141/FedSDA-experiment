# 要求 revision 2

## 1. 初期登録の組立

- 1.1 When 学習済みの採用候補と負の一時IDが渡されたとき, the registration operation shall 現在の学習帰属IDのモデルが保有されていればその共有特徴抽出部を、保有されていなければ保有一覧先頭のモデルの共有特徴抽出部を反映先に選び、候補の共有部の値を反映先へ反映し、候補を反映先へ接続し、候補の個別optimizerをresetする。
- 1.2 The registration operation shall 渡された特徴とラベルに対する候補の標本別有界損失を一回のforwardで評価し、全体と出現クラスの初期損失統計を生成する。損失と統計の値は、共有反映後に接続済みの候補で評価した値と一致する。
- 1.3 When 登録を完了したとき, the registration operation shall 候補と個別optimizer管理器を一時IDで学習状態一覧の末尾へ登録し、初期統計を同IDへ設定し、接続後の候補と同じ値の独立したparameter snapshotを同IDと1以上の待機ラウンド数で送信保留へ登録している。別IDの既存の送信保留は置換する。
- 1.4 The registration operation shall 現在の学習帰属ID、学習標本、評価標本、学習/割当計数、既存保有モデルの概念固有parameter/grad/個別optimizer、既存モデルの統計、共有optimizerの蓄積stateを変更せず、乱数を消費しない。共有部の値の変更は、同じ共有部を参照する全保有モデルへ及ぶ。

## 2. 呼出し契約

- 2.1 If 一時IDがbuiltin int以外または非負なら, the registration operation shall TypeErrorまたはValueErrorを送出し、どの状態も変更しない。bool/int派生型/暗黙変換を受理しない。
- 2.2 If 一時IDが学習状態一覧に保有済み、統計storeに登録済み、または既存の送信保留の対応IDと同じなら, the registration operation shall ValueErrorを送出し、どの状態も変更しない。
- 2.3 If 待機ラウンド数がbuiltin int以外または1未満なら, the registration operation shall TypeErrorまたはValueErrorを送出し、どの状態も変更しない。
- 2.4 If 渡されたownerが対象の具体型でないなら, the registration operation shall TypeErrorを送出し、どの状態も変更しない。
- 2.5 If 保有モデルが一つもないなら, the registration operation shall 反映先を選べないためLookupErrorを送出し、どの状態も変更しない。
- 2.6 If 候補・個別optimizer管理器・反映先・特徴・ラベルのいずれかが既存の共有反映・損失評価・初期統計・parameter snapshotの検証で拒否されるなら, the registration operation shall その例外を伝え、共有部の値、候補の接続先と個別optimizer、学習状態一覧、統計、送信保留のどれも変更しない。
- 2.7 The registration operation shall 2.1〜2.6の検証と数値生成を全て終えてから状態を変更する。検証済み入力に対する既存owner APIの順次更新であり、private状態の改変・並行更新・メモリ不足等による途中例外へのrollbackは提供しない。

## 3. 境界と移植

- 3.1 The registration operation shall 各ownerを借用して既存APIで更新し、自身の永続状態を持たない。一時IDの採番、候補の生成/学習/採否、学習量の帰属、保留標本の追加、現在の学習帰属IDの切替えと通知、計算量診断の記録、送受信、予測重みの変更を行わない。
- 3.2 When 有効な同じ状態へ登録を適用したとき, the new registration operation shall 実旧登録処理と旧FedSDAの待機設定による共有部の値、保有一覧の順序、候補の出力、初期統計の全field、送信保留のparameter値と待機・ready状態を維持する。登録後の共同学習が実旧と同じ数値とoptimizer stateで継続できるようにする。
