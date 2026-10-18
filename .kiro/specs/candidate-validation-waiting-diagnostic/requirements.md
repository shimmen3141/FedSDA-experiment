# Requirements Document

## Project Description
研究者は候補検証なしの適応利得を残し、低下と通信増を簡単な制御で抑えたい。現在は待機・凍結・選別が一緒に変わるため、その切り分け用に、同じ検証待機の後で必ず採用する診断対照を実行できるようにする。

## Boundary Context
対象は診断専用ランナーと証拠の保存。製品src・既定・既存記録形式・goldenは変更しない。共有ゲートGの実装と採否は対象外。手元の全規模NN実験は環境が使える場合に行い、サーバコマンドを用意できることまでを本specの完了条件とする。

## Requirements
### Requirement 1: 待機と選別を分ける対照
1. The diagnostic shall 現行検証ありV、現行即時採用O、検証待機後強制採用Dを区別して実行する。
2. While Dの検証標本が規定10件未満である, the diagnostic shall Vと同じ候補凍結・参照評価・保留標本処理・同期の経路を使う。
3. When Dの規定件数が揃う, the diagnostic shall 元の評価を計算した後、履歴/二分区間/再利用の判定にかかわらず候補を採用し、通常の採用処理につなぐ。
4. If Dの候補がstream終端まで未完了である, the diagnostic shall 現行Vと同じ終端回収を行い、未完了候補を強制採用しない。
5. The diagnostic shall 候補作成前の既存モデル再利用判断をV/O/Dの各現行経路に従って維持する。

### Requirement 2: 診断として誤読されない記録
1. The diagnostic shall 強制採用を通常合格と異なる理由で記録し、元の合否・比較損失・参照IDも保持する。
2. The diagnostic shall 実験条件・mode・purpose・source commit/dirty・source/script hash・環境・処理標本数・実際の指標とNPZを診断専用形式へ保存する。
3. The diagnostic shall 全modeで既存のrun設定を保持し、その上に適用した介入を独立に明示する。

### Requirement 3: 再現性と安全な実行
1. The diagnostic shall V/Oにおいて元実行と全観測配列・指標・終了時RNGが一致する。
2. The diagnostic shall Dの評価介入で追加の乱数を消費せず、元の評価値を変更しない。
3. If 実行が失敗する, the diagnostic shall 適用中の差替えを復元し、不完全な出力を完了結果として扱わない。
4. If 引数が不正または対象runの出力が既存である, the diagnostic shall 学習前に拒否し、既存結果を上書きしない。
5. When 計画表示が指定される, the diagnostic shall run格子を表示するだけで、結果ディレクトリや学習状態を作らない。
6. The diagnostic shall workerごとに独立processで1runのみを処理し、スレッドを各1に制限する。
