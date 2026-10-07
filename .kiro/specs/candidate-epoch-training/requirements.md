# 候補のエポック学習 — 要求 revision2

## 目的と範囲

生成済み独立候補の初期学習を、グローバル設定やclient登録と分離する。旧正常経路の値・順序・乱数消費を維持する。固定エポック・検証損失早期停止・学習省略を含み、参照を一緒に学習する方式は含めない。

## 1. 学習方式

- 1.1 When 固定エポック学習を指定したとき、the Candidate Training shall 区間全体を各エポックで並べ替え、指定batch上限で末尾の小batchも学習する。
- 1.2 When 検証損失早期停止を指定し学習標本を1件以上確保できるとき、the Candidate Training shall 区間を一回だけ無作為分割し、前回採用した検証損失から指定閾値を厳密に超えて減少した場合だけ改善として採用し、閾値ちょうどを含む非改善の連続回数により指定上限以内で停止する。
- 1.3 If 早期停止の分割で学習標本を1件確保できないとき、the Candidate Training shall 区間全体による固定エポック学習へ切り替える。
- 1.4 When 学習省略を指定したとき、the Candidate Training shall 候補の値・勾配・optimizer・乱数を変更せず、学習量を0として返す。

## 2. 旧挙動の保全と学習量

- 2.1 When 正常入力で学習したとき、the Candidate Training shall 旧実装と同じ終端parameter・勾配・optimizer状態・乱数状態を得て、入力区間を変更しない。
- 2.2 When 早期停止の学習を終了したとき、the Candidate Training shall 指定閾値を厳密に超える改善として最後に採用した時点のparameterだけを復元し、optimizerと勾配は最後に実行した更新の状態を保持する。改善採用がなければ学習前のparameterを復元する。
- 2.3 When 学習を返すとき、the Candidate Training shall 実行済みエポック数・延べ学習標本数・parameter更新回数・延べ検証標本数を返す。最良値への復元で計数を巻き戻さない。

## 3. 契約と境界

- 3.1 If 学習設定・分類器・専用optimizerの結合・区間の型/shape/値・必要な勾配有効条件が不正なとき、the Candidate Training shall 学習更新と乱数消費の前に拒否する。正の標本数、CPU float32、非空共有parameter、分類器に適合する観測クラス値を受理条件とする。
- 3.2 While 本機能を実行するとき、the Candidate Training shall 候補生成、初期値選択、登録、ID採番、参照固定、session開始、外部モデルへの共有接続、通信、外部診断owner更新を行わない。
- 3.3 When 本specを完了とするとき、the Candidate Training shall 実旧との方式別・小区間・端数batch・早期停止・復元の対照、生成部品からの学習接続、依存境界、基準環境の全回帰を証拠として残す。新全体runの完成は主張しない。
