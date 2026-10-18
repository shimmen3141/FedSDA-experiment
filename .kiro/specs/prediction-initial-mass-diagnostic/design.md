# Design Document

## Overview / Goals
新しいIDに一様な大きな初期質量を与えることが、検証なしの低下原因か調べる。O→Rで既存比率保持、R→Pで新ID初期質量を切り分ける。Pの有効性は未確定。共有部への干渉対策や共有ゲートGは対象外。

## Boundary Commitments / Architecture
専用run.pyが通常execute_fedsda_experiment_runを呼び、R/Pのrun中だけFixedShareWeightController._synchronize_model_poolをcontextmanagerで差し替える。finallyで元へ復元。専用checks.pyは明示pytestのみ。専用ディレクトリを取り除けばコード側の変更はなくなる。共通登録・除外設定・新製品オプションは追加しない。
既存Dランナーの保存/計画/worker構造を専用run.pyに複製し独立させる。Dコードには依存せず変更もしない。ルートは__file__.parents[2]。公開関数追加はなくCLI moduleのみ。

## Components and Behavioral Contract
元の同期を呼ぶ前に旧重みを独立dictへ退避し、この旧状態から残存/追加集合と質量を求める。旧重みw、現在ID集合I、残存S、追加N、n=|I|、k=|N|、s=sum(w_i for i in S)、既存Tを使う。
- N非空かつS非空かつs>0の時だけ介入。まず元の同期を1回呼び（集合reset計数と分散0を維持）、その後重みだけ置き換える。
- R: 各新IDに1/n。既存ID iへ(1-k/n)*w_i/s。
- P: 各新IDに1/(n*T)。既存ID iへ(1-k/(n*T))*w_i/s。新しい自由パラメータなし。Tは既存設定の有効域（2以上）に従う。
- 初回・同一集合・削除のみ・全置換・残存質量0は元の処理。現在ID昇順、有限非負で総和1。旧重みと新重みを独立dictでeventへ保存する。生のobject IDは保存しない。
- 通常のFixed-Share再配分はそのままなので、Pの初期抑制が次の損失観測後どれだけ残るかも分析対象。集約でclearされる時は一様から再較正し、この介入の対象外。
- Vはfedsda、O/R/Pはfedsda-without-candidate-validation。混合予測・FIFO/Tなど既存設定をそのまま使う。

## Data Models / Error Handling
format=prediction-initial-mass-diagnostic-v1。D診断と同じpayload項目でintervention_eventsだけを重み同期eventとし、元の候補評価フィールドは持たない。eventはold_weights,new_weights,model_ids,new_model_ids,time_scale,mode。NPZは既存全記録。source/script hashを親計画・worker開始・終了で照合。RNG全状態fingerprintは既存Dと同じ方法。既存runディレクトリは空でも拒否、run単位mkdir(exist_ok=False)、NPZ→JSON完了印。自動再開なし。

## Testing Strategy / Requirements Traceability
1: 専用テストで追加/複数追加/削除/置換/0残存/T下限2/集合不変、ラベル不要、計数/分散、RNGと例外時復元を確認。既存比率の違いを含む例でRとPを区別する。
2: 専用テストで計画のread-only・予約競合・不正値と保存来歴、V/O直接対照。共通src/設定/CI/freshのdiff空と通常収集対象外の命名を確認。
3: 小規模24run（6条件×seed0×4方式、1000/client）。大規模は6条件×seed0..2×4=72runを計画表示で確認してコマンド提供。研究室サーバは主担当から起動しない。学習不変はNPZの予測/重み診断を除く全配列、最終state dict、通信/学習/適応指標を比較し許容差で隠さない。真の概念は診断だけに使う。

### 実験条件と最終パラメータ証拠
6条件はMNIST2:100、MNIST2:200、SEA4:50、MNIST4:500、CIRCLE2:200、SINE2:500。全て10clients、pilot1000/client・seed0・4方式、本規模5000/client・seed0..2・4方式。既定FIFO30/T120は設定から取得し保存する。
最終グローバルパラメータはserver_operations.owners.global_model_repositoryから取得し、各model IDの各state_dict keyについてdtype/shape/連続CPU byteのSHA256をfinal_global_parametersフィールドへ保存する。これで別processのO/R/Pの最終学習状態を実値由来のdigestで照合する。全体hashだけで隠さずmodel/key単位とする。Dのpayloadにはない追加の研究用fieldであり既存記録形式は変えない。
