# リファクタリングの進行

再開時はまず[短い再開案内](resume.md)を読む。承認・進捗の正本は各specのspec.json/tasks.md。

## 全体方針

旧基準`748c3aa`と同じGit履歴の`refactor/architecture`ブランチで、新APIへ移行する。
クリーンアーキテクチャの依存方向・差し替え境界・機能別配置を採用する。
詳細の正本は`docs/research/refactoring-policy.md`。新srcのパッケージ境界と機能ごとの設定型を段階的に実装している。

## specの候補と依存順

1. `configuration-foundation`: 初回は最終構成に必要な型付き設定と検証。選択肢、preset、保存表現は必要になる段階で追加する。
2. `single-run-execution`: SINEデータ供給と単一runの実行順序。12実行タスクを完了し、2026-10-03の統合検証はGO。承認・検証証拠は同specのspec.jsonとreview.mdを参照する。
3. 最終FedSDAの予測・割当・監視・候補検証: 最初は`fixed-share-prediction-weights`で予測重み状態と更新を独立移植する。続いてモデル出力混合、ClassESR監視、候補の将来損失判定、FIFO帰属・client調整を責務別に仕様化し、1・2へ接続する。
4. モデル・学習・サーバ同期・統合: 3の境界に沿って移植する。
5. 評価・新成果物・golden比較と必要なbaseline・ablation: 各移植単位から段階的に接続する。
6. 新CLI・掃引・文書を仕上げ、新ブランチから参照用旧構成を除く。

3の最初のspecは`fixed-share-prediction-weights`。続く`weighted-class-probability-prediction`も3実装タスクを完了し、Lunaによる最終統合判定GO。すでに得られた出力の確率化・混合・クラス判定・観測後損失に限定する。先の関数・変数を全て今決めることはせず、担当単位の実装前に一覧化する。
機能境界と状態所有者を先に整理し、最小設定を小規模な実行経路へ接続する。
汎用の設定基盤を全て完成させてから手法を移植する順序にはしない。

## 現在の状態

- `candidate-epoch-training`を完了。要求revision2・設計revision1・命名revision7・tasks revision1、全5task承認、別fresh GPT-6 Lunaのfeature最終GO。固定エポック/検証損失早期停止/省略、0epoch/小区間/端数batch、parameterのみ復元、全状態/RNG/学習量を実旧と照合した。生成→学習→継続更新とexact依存境界も検証。commit66ac055で全6868passed/3skipped/2warnings、旧11・最終3goldenと品質検査成功。次はsession開始の組立。入口はresume.md、承認・進捗は同specのspec.json/tasks.mdが正本。

- `candidate-classifier-construction`を完了。要求/設計revision2・命名revision4・tasks revision2、全3tasks承認、別GPT-6 Lunaセッションでfeature最終GO。独立候補と専用optimizer生成、初期snapshot選択→3batch更新の実旧対照、exact依存境界を検証した。検証commit `ceb4336`で全6412passed/3skipped/2warnings、旧11・最終3goldenと品質検査成功。警報区間のepoch学習・early stopping・session開始、新client/全体runは後続。次は候補学習の仕様化、入口はresume.md。

- 警報時点の参照モデルの固定は`../specs/post-alarm-reference-model-fixation/README.md`。主担当Claude Code。要求revision1・設計revision1・命名revision2・tasks revision1・全3tasks承認/完了、別feature最終GO（3回目。1回目は基準の誤読、2回目は元checkoutの取り違えによるNO-GOで、いずれも実装・testへの指摘ではない）。保有モデルごとに同じ値の独立した参照分類器を一覧の順に生成し、全体統計が2件以上のモデルの履歴平均損失（平均0を含む）を返す。値の検証を全モデルぶん先に行い、拒否時はtorch乱数を消費しない。実旧_snapshot_reference_modelsと6条件で処理後のtorch乱数状態・参照IDの順・全値・出力を照合、履歴平均8条件を実旧session開始と照合、拒否10条件、session開始→観測→確定の照合6条件、12条件で学習継続中も参照が固定されたままであることを確認。実装前にREDを実行し、実装差し替え11種の検出も確認。対象＋AST1059/全6355passed・3skipped・1既存warning（主担当実測＋JUnit）、fresh新CPU/品質/固定旧golden/8要件成功。次は候補の生成と警報区間での学習、session開始の組立。入口はresume.md。

- 警報後の候補検証標本の観測は`../specs/post-alarm-candidate-validation-sample-observation/README.md`。主担当Claude Code。要求revision1・設計revision2・命名revision2・tasks revision1・全3tasks承認/完了、別feature最終GO。検証標本1件について候補と警報時点で固定した参照分類器の損失を評価し、既存の損失収集へ1回で追加して規定件数への到達を返す。1標本契約はforwardより前に検査し、到達後の評価・確定は呼ばない。実旧_observe_forward_validationと4条件で損失列を照合、拒否24条件、到達後の評価→確定まで含めた状態一致6条件、12条件の学習継続。実装前にREDを実行し、実装差し替え10種の検出も確認。対象＋AST1028/全6277passed・3skipped・1既存warning（主担当実測＋JUnit）、fresh新CPU/品質/固定旧golden/9要件成功。参照も学習させる方針（旧shadow_tournament）は2026-10-07のユーザー判断で当面不要（goldenにも値なし）。次は候補検証sessionの開始（参照分類器の固定、候補の生成と学習。モデル生成の乱数消費順に注意）。入口はresume.md。

- 警報後の候補検証の確定は`../specs/post-alarm-candidate-validation-resolution/README.md`。主担当Claude Code。要求revision1・設計revision2・命名revision2・tasks revision2・全3tasks承認/完了、別feature最終GO。評価結果record（採用の可否・再利用可能な参照）から、採用の組立、吸収＋現在の学習帰属ID切替え、吸収だけ、のいずれかを選んで適用し、結果種別4値・帰属先・変更記録を返す。再利用は吸収→切替えの順にして拒否時に現在IDを変えない。実旧_finalize_forward_validationの4分岐と16条件で全状態・旧action・旧戻り値・変更通知引数を照合、拒否75条件、24条件の確定後学習。実装前にREDを実行し、実装差し替え10種の検出も確認。対象＋AST1071/全6200passed・3skipped・1既存warning（主担当実測＋JUnit）、fresh新CPU/品質/固定旧golden/11要件成功。次は候補検証sessionの進行（損失評価→収集→評価→確定）と記録/通知。入口はresume.md。

- 帰属確定標本の吸収は`../specs/assigned-training-sample-absorption/README.md`。主担当Claude Code。要求revision1・設計revision1・命名revision3・tasks revision1・全3tasks承認/完了、別実Luna feature GO（2回目。1回目は最終レビュー記録の未記載とレビュー担当の自己同定不能を理由にNO-GO）。外側runtimeの状態なし関数で、保有モデルへの標本追加・割当概念計数・標本ごとの損失評価・全体/クラス別統計更新を組み立てた。全標本の検証と損失評価を状態変更より前に置き、拒否時は先行標本を含め全状態が不変。実旧_absorb_into_storeと72条件、実旧確定処理の非採用3分岐6条件、拒否35条件、12条件の吸収後学習を照合。Task1はREDを実装より前に実行しない手順逸脱があり、実装差し替え8種の検出で代替した。対象＋AST1036/全6038passed・3skipped・1既存warning（主担当実測＋JUnit）、fresh新CPU/品質/固定旧golden/12要件成功。旧の契約外入力での部分更新をLEGACY-015へ記録。次は確定処理の判断と分岐の組立。入口はresume.md。

- 採用候補のローカル採用は`../specs/adopted-candidate-local-adoption/README.md`。主担当Claude Code。要求revision2・設計revision2・命名revision2・tasks revision2・全3tasks承認/完了、別実Luna feature GO。外側runtimeの状態なし関数で、採番owner・初期ローカル登録・学習計数・学習標本store・現在の学習帰属IDを組み立てた。登録の検証が通るまで採番を確定せず、拒否時は採番次値を含む全状態が不変。保留標本は追加だけで統計・割当概念計数を変えない（旧採用分岐と同じ）。実旧_finalize_forward_validationの採用分岐と48＋2条件で全状態を照合、拒否40条件、class2/4×3optimizer×共有更新有無の12条件で採用前学習→採用→学習→正式ID確認→学習を実旧へ照合。対象＋AST977/全5876passed・3skipped・1既存warning（主担当実測＋JUnit）、fresh新CPU/品質/固定旧golden/11要件成功。正常経路の不具合は観測していないが、契約外入力での部分更新（LEGACY-012/013）と、採用時の保留標本が割当概念計数・統計へ反映されない非対称（LEGACY-014、意図未確認）を記録した。次は確定処理の残り（棄却・再利用・維持分岐と記録/通知）と候補sessionの開始。入口はresume.md。

- 一時モデルIDの採番は`../specs/temporary-model-id-allocation/README.md`。主担当Claude Code。要求revision1・設計revision2・命名revision3・tasks revision2・全2tasks承認/完了、別実Luna feature GO。次に採番する負IDを一つ所有するownerを学習層へ追加し、初期値-100−client_idと単調減少の採番を移植。実旧BaseClientの実初期化/実採番と16条件で採番列を照合、拒否8、import全拒否のexact guardと注入契約24、stdlib単独起動、採番IDでの連続登録2条件。対象＋AST861/全5731passed・3skipped・1既存warning（主担当実測＋JUnit）、品質/固定旧golden/6要件成功。旧サーバの回収経路では異なるclientの同値一時IDは衝突しないことを確認（診断・成果物側は調査範囲外）。次は候補採用時の接続。入口はresume.md。

- 採用候補の初期ローカル登録は`../specs/adopted-candidate-initial-local-registration/README.md`。主担当Claude Code。要求revision2・設計revision2・命名revision3・tasks revision3・全3tasks承認/完了、別実Luna feature GO（全pytestは主担当実測＋JUnit照合、基準はagent-handoff.md）。外側runtimeの状態なし関数で、現在ID優先/一覧先頭の反映先選択、損失・初期統計・snapshotの事前生成、共有反映→学習状態一覧→統計→送信保留の順次更新を組み立てた。全検証と数値生成を状態変更より前に置き、拒否時は共有部・接続・optimizerを含む全状態が不変。実旧対照48＋統計標本8＋optimizer3条件、拒否47条件、class2/4×3optimizer×共有更新有無の12条件で登録前学習→登録→学習→正式ID確認→学習の全数値/optimizer/RNGを実旧へ照合。対象＋AST928/全5678passed・3skipped・1既存warning、fresh新CPU/品質/固定旧golden/13要件成功。新たな旧正常不具合は観測していない。次は候補session終了時の採用接続（採番・計数・標本追加・現在ID切替えと通知）。通信・新client/全体runは後続、入口はresume.md。

- 保有モデルの正式登録確認は`../specs/held-model-registration-confirmation/README.md`。要求/設計revision1・命名revision2・全3tasks承認/完了、別実Luna feature GO。外側runtimeの状態なし関数で7ownerのID上書き/計数加算/current変更/pending解除を組み立てた。実旧保有32・非負3条件、全入力拒否/欠落model事前拒否/順序/保持を確認。class2/4×3optimizer×共有更新有無の12条件3共同更新で確認後の全数値/optimizer/count/RNGを実旧へ照合。対象＋AST864/全5518passed・3skipped・1既存warning、fresh新CPU/品質/固定旧golden/9要件成功。新たな旧正常不具合は観測していない。次は採用済みモデルの初期ローカル登録。欠落model復元・通信・新client/全体runは後続、入口はresume.md。

- 現在の学習帰属モデルIDは`../specs/current-training-model-assignment/README.md`。要求revision3・設計revision1・命名revision2・tasks revision2・全3tasks承認/完了、別実Luna feature GO。単一ID owner/変更record/全入力事前拒否/一段mapを移植し、理由別通知は上位へ残した。実旧ローカル16・server28条件、登録確認と計数移管3条件を対照。対象＋AST819/全5398 passed・3 skipped・1既存warning、stdlib単独/品質/固定旧golden/全7要件を確認。新たな旧正常不具合は観測していない。root探索のRuff panicと回避をdevelopment finding/品質手順へ記録。次は正式ローカル登録の組立。全登録・新client/全体runは後続、読む入口はresume.md。

- モデル別学習・割当件数は`../specs/model-training-and-assignment-counts/README.md`。要求/設計revision1・命名revision2・全3tasks承認/完了、別実Luna feature GO。3独立計数の加算・copy・異ID加算移管・一回ID対応合計を移植し、真concept診断と個別parameter stepの意味を明示。実旧193単体条件と12NN条件3学習/ID移管後の全計数・batch/数値/optimizer/Randomを照合。対象＋AST931/全5305 passed・3 skipped・1既存warning、stdlib単独/fresh新CPU/品質/固定旧golden/全9条件を検証。旧同一負ID通知の件数倍増・concept消失をLEGACY011へ記録し、新APIは同IDを事前拒否する。通常経路/過去成果への影響は未確認。次は現在帰属IDと正式ローカル登録組立の不足を確認する。全登録・新client/全体runは後続、読む入口はresume.md。

- モデル別評価標本の保持は`../specs/model-evaluation-sample-storage/README.md`。要求/設計/命名revision1・全3tasks承認/完了、別実Luna feature GO。評価recordと一storeを新evaluation層へ追加し、明示Randomによる抽出追加/末尾容量保持、単一IDの先上書き、一回ID連結/超過列だけの再抽出を移植。実旧290保持条件とclass2/4の2先容量超過→損失接続で順序/借用参照/Random/loss一致を確認。対象＋AST1000/全5082 passed・3 skipped・1既存warning、fresh新CPU/品質/固定旧golden/全11条件を検証。新たな旧正常不具合は観測していない。次はモデル別加算counter/現在帰属IDの不足を確認する。評価fallback選択・全登録・新client/全体runは後続、読む入口はresume.mdを参照。

- モデル学習標本の単一ID付替えは`../specs/model-training-sample-id-reassignment/README.md`。要求/設計/命名revision1・全3tasks承認/完了、別実Luna feature GO。既存storeへ一APIを追加し、標本列をpop上書きして列順/空/重複/借用参照と取得済みsnapshotを保持する。実旧confirmの順序・上書き/欠落/同ID、両ID拒否とpayload非検査、12条件3更新の抽出/全数値/optimizer/Random一致を確認。対象＋AST768/全4746 passed・3 skipped・1既存warning、fresh新CPU/品質/固定旧golden/全8条件を検証。新たな旧正常不具合は観測していない。評価用stored_data/容量・加算counter・現在帰属ID・正式登録全体・client/通信/新全体runは後続。次の候補と読む入口はresume.mdを参照。

- 保有モデル学習状態の単一ID付替えは`../specs/held-model-training-state-id-reassignment/README.md`。要求/設計revision1・命名revision3・全3tasks承認/完了、別実Luna feature GO。既存registryへ一APIを追加し、新wrapperだけIDを変更、NN/parameter/grad/共有参照/個別optimizer管理器と蓄積stateを保持する。実旧pop順/先上書き/欠落/同ID・古いrecord/binding・両ID拒否、12条件3共同更新と統計上位接続で全数値/状態一致を確認。対象＋AST747/全4642 passed・3 skipped・1既存warning、fresh新CPU/品質/固定旧golden/全8条件を検証。新たな旧正常不具合は観測していない。標本/counterの付替え・現在帰属ID・正式登録全体・client/通信/新全体runは後続。

- 損失統計の単一モデルID付替えは`../specs/loss-statistics-model-id-reassignment/README.md`。要求/設計/命名revision1・全3tasks承認/完了、別Luna feature GO。既存storeへ一APIを追加し、実旧正式登録のpop代入/既存先上書き位置維持/新先・同ID末尾/元欠落no-opを照合。両ID事前検査/取得値独立/変更先更新と4条件の現在統計・固定snapshot・保留不変/明示解除接続を確認。対象＋AST731/全4559 passed・3 skipped・1既存warning、fresh新CPU/品質/固定旧golden/全8条件を検証。新たな旧正常不具合は観測していない。モデルregistry/標本/counterのID付替え・現在帰属ID管理・正式登録全体・新client/全体runは後続。

- 新規モデルの送信保留は`../specs/pending-model-upload/README.md`。要求/設計revision1・命名revision2・全3tasksをLuna承認/完了、別feature統合GO。一保留枠/生成済みsnapshot借用/対応ID/正のラウンド待機/非消費取得/明示解除を移植。実旧9状態列と6実NN/統計更新接続で、登録時モデル値固定と現在統計全field/readinessをexact照合。対象＋AST731/全4492 passed・3 skipped・1既存warning、fresh新CPU/品質/固定旧golden/全11要件を確認。旧解除後の無効counterは空の有効残数0として明示対応し、送信時期は維持。新たな旧正常不具合は観測していない。統計/モデル所有・正式登録/ID対応・実送信・client進行/全体runは後続。

- 分類器parameter snapshotは`../specs/classifier-parameter-snapshot/README.md`。要求/設計revision1・命名revision3・全3tasksをLuna承認/完了、別feature統合GO。検証後のnative順全値detached独立copyを実装。実旧class2/4/10のget_paramsと12条件3学習後のsnapshot→既存initializer→native復元/予測/optimizer保持をexact照合。対象＋AST666/全4399 passed・3 skipped・1既存warning、fresh新CPU/品質/固定旧golden/全8要件を確認。新たな旧正常不具合は観測していない。送信保留状態・統計/標本登録・ID対応・新client/全体runは後続。

- 準備済み分類器の標本別有界損失は`../specs/classifier-per-sample-bounded-loss-evaluation/README.md`。要求revision2・設計revision2・命名revision1と全3tasksをLuna承認・完了、feature最終GO。一forwardの標本別有界損失と事前/出力検証を移植し、実旧24条件の損失と12条件3学習後の初期統計全fieldをexact照合。対象＋AST701/全4345 passed・3 skipped・1既存warning、fresh新CPU/品質/固定旧差分/全11要件を確認。新testのdevice context漏れはscoped contextへ修正しdevelopment findingへ記録。新たな旧正常経路の不具合は観測していない。状態・登録・学習・採否・送信・新client/全体runの所有を含まない。

- 保有モデルの学習状態管理は`../specs/held-model-training-state-registry/README.md`。要求revision3・設計/命名revision1・全3taskはLuna APPROVED完了、最終feature GO。ID別NN/個別optimizer管理器を初出順で保持し、同ID置換と現在binding取得を移植した。実旧登録順と12条件3stepの全数値/状態/Random一致、対象＋AST617/全4229 passed・3 skipped・1既存warning、fresh新CPU/品質/固定旧差分/全11要件確認。依存guardのmodule import受理を修正し既存guardの見直し候補をdevelopment-findingsへ記録。新たな旧正常経路の不具合は観測していない。統計/標本/送信snapshot/ID対応/正式登録全体/新client・全体runは後続。

- 採用候補の共有学習反映は`../specs/adopted-candidate-shared-feature-integration/README.md`。要求/設計/命名revision1・全3tasks APPROVED完了、Luna最終feature GO。値反映→候補接続→個別resetを分離し、既存共有parameter/grad参照と共有optimizer stateを維持する。実旧12NN条件×3stepの全loss/値/grad/state一致、対象37/AST込み592、全4167 passed/3 skipped/1既存warning、品質/fresh新CPU/固定旧差分/全9要件trace確認。候補採否・登録/ID/統計・新client/全体runは後続。新たな旧正常系不具合は観測していない。

- 保有モデル全体の共有再接続は`../specs/held-model-shared-feature-reconnection/README.md`。全3tasks完了・設計revision2/命名revision2・Luna最終feature GO、対象49/AST込み580、全4106 passed/3 skipped/1既存warning。品質/fresh新CPU/旧固定差分/全11要件traceを確認。共有元選択/source保持/non-source接続と個別reset/現在owner結果を外側責務として移植。実旧12NN条件×3stepの全loss/NN値/grad/optimizer state一致、旧共有owner非流用/不変を確認。同期parameterロード・候補採用・モデル一覧所有・新client/全体runは後続。新たな旧正常系不具合は観測していない。

- 共有特徴抽出部への再接続は`../specs/shared-feature-extractor-attachment/README.md`。全3tasks完了・命名revision3・Luna最終feature GO、対象33/AST込み540、全4033 passed/3 skipped/1既存warning。品質/fresh新CPU/旧固定差分/全7条件traceを確認。参照交換/接続検証と上位のoptimizer操作を分離する。実旧12条件×3stepの全loss/NN値/grad/optimizer state一致、旧owner非流用/不変を確認。共有元選択・同期parameterロード・候補登録・モデル一覧の所有・新client/全体runは後続。新たな旧正常系不具合は観測していない。

- optimizer状態の所有と明示リセットは`../specs/parameter-optimizer-state/README.md`。全3tasks完了、命名revision3/Luna最終feature GO。固定parameter列/設定を借用し現在optimizerと成功後交換を所有、値/gradと旧借用binding保持。実旧reset12条件・NN24条件×3stepを照合、対象47/AST込み554、全4000 passed/3 skipped/1既存warning、品質/fresh新CPU smoke/旧固定差分確認。全9要件/hash/採否は対象specが正本。モデル所有/共有接続/正式登録/reset時機/新client/全体runは後続。新たな旧正常系不具合は観測していない。

- モデル別学習標本の保持は`../specs/model-training-sample-storage/README.md`。全3tasks完了、命名revision1/Luna最終feature GO。追加順/空列/重複/借用payload/snapshot構造分離/サーバID一回対応を独立移植。実旧保存16条件・抽出48条件×3反復を照合、対象94/AST込み581、全3933 passed/3 skipped/1既存warning、品質/fresh新CPU smoke/旧固定差分確認。全10要件/hash/採否は対象specが正本。正式登録pop/上書き、モデル/optimizer所有・同期、統計/評価store、新client/全体runは後続。新たな旧正常系不具合は観測していない。

- ローカル学習要求の保留と実行回数は`../specs/local-training-request-scheduling/README.md`。全3tasks完了・命名revision1/Luna最終統合GO。要求counter/間隔/試行予算/正常終了の明示確認を独立移植。実旧9条件の要求列と32条件の実NN反復接続でcounter/予算/順序/loss/parameter/grad/optimizer/RNGを照合。対象70/AST込み539、全3821 passed/3 skipped/1既存warning、stdlib/new CPU4-step smokeを主担当・Lunaが確認。旧source/golden変更なし、全11要件/hash/採否は対象specが正本。完全run設定/CLI、要求・警報/ラウンドflush位置、標本/optimizer所有、通信/診断・新client/全体runは後続。新たな旧正常不具合は観測していない。

- 保有モデルの共同学習反復は`../specs/held-model-joint-training-iterations/README.md`。全3tasks完了・命名revision2/Luna最終統合GO。借用モデルID対応と毎回新抽出→共同更新を接続し、実旧72条件で全batch/loss/parameter/grad/optimizer/RNGをexact照合。対象94/AST込み533、全3721 passed/3 skipped/1既存warning、主担当・Lunaのfresh CPU4回Adam更新/旧非import成功。0/skip/拒否・後続失敗時の先行更新保持も確認。全12要件・hash・採否の正本は対象spec。optimizer/標本所有・回数算出/pending/interval・候補/同期/診断counter・PCGrad・新全体runは後続。新たな旧正常不具合は観測していない。

- 保有モデルの学習バッチ抽出は`../specs/held-model-training-batch-sampling/README.md`。全4tasks完了・Luna最終統合GO。順序付きモデル別観測列から保有/件数で参加選別し、借用Randomで復元なし抽出とcatを実装。対象56/AST込み459、全3591 passed/3 skipped/1既存warning、主担当/LunaのfreshCPU抽出→共同更新smoke成功。FIFO位置解決/IDからNN・optimizer対応はtest-onlyで実旧12条件×3stepへ完全照合した。全12要件・hash・採否・実測は対象specへ記録。標本ストア所有/追加/帰属/ID統合・反復・optimizer所有/reset・新全体runは後続。新たな旧正常不具合は観測していない。

- 一回共同更新は`../specs/joint-model-parameter-update/README.md`。全3tasks完了・Luna最終統合GO。確定済み参加batch列から共有forward一回・標本数加重BCE/CE・backward一回・共有step→個別step入力順を移植。対象77/AST込み438、全3493 passed/3 skipped/1既存warning、主担当/LunaのfreshCPU smoke成功。空共有はoptimizer=Noneで個別学習を独立検証し、旧LEGACY010は未修正/通常過去影響未確認。参加抽出は上記specで独立移植した。反復・optimizer所有/reset・候補進行・通信/診断counter・PCGrad・新全体runは後続。全17要件・hash・採否・実測の正本は対象spec.json/review.md。

- optimizer設定/生成は`../specs/parameter-optimizer-construction/README.md`。全3tasks完了・Luna最終統合GO。Adam standard/AMSGradとlrのみSGDを別型にし、明示Parameter列へ標準optimizerを生成。実旧groups/state/3step全値を照合、新NN共有/個別tuple接続、346件・全3364 passed/3 skipped・fresh CPU smokeを検証。9条件・hash・採否/実測の正本は対象spec。LEGACY010へ空共有optimizer生成失敗を記録、旧未修正/通常過去影響未確認。単回共同更新は上記specで完了。共有optimizer所有/reset・学習反復・候補進行・新全体runは後続。

- 最終モデル構造の移植は`../specs/residual-adapter-model-architecture/README.md`。全3tasksを完了しLuna最終統合GO。共有特徴抽出/非線形残差/classifierのCPU32生成・forward、実旧state/RNG/gradient、test-only確率/snapshot接続を実装。対象＋AST319件、全tests3284 passed/3 skipped、CPU fresh smokeを検証した。承認hash・12条件・実測・採否は対象specに記録。追加の旧不具合は観測していない。学習・optimizer・共有付替え・候補進行・新全体runは後続。

- worktreeとcc-sdd導入: 完了。
- アーキテクチャ規約・命名レビュー手順: 作成済み。
- 要求・設計・初回task: 人間承認済み。初回13タスクに既存成果の改名task 9.1を追加した。
- 設定基盤の初回承認範囲と後続未完了範囲: `../specs/configuration-foundation/spec.json`が正本。
- 初回範囲: 基礎例外、各機能の固定条件、値・組合せ検証、不変な部分集約型。実験実行や完全な設定型は含めない。
- 正本一覧は対象specのREADME。命名再検討表は候補・履歴であり、実装する正式名はnaming.mdのみ。
- 単一run基盤: `../specs/single-run-execution/README.md`。12実行タスクを完了。旧SINEとの準備後データ・乱数・処理順の照合と既存goldenを含む全テストを確認した。新しい学習・判断処理の研究指標は未移植。
- 予測重みの移植は`../specs/fixed-share-prediction-weights/README.md`。4実装タスクはLuna承認・検証済み、最終統合判定はGO。全testsは2172 passed / 3 skipped、旧production・golden差分なし。承認・feature GOの正本は同spec.json。モデル混合・学習と全体runはこの部品の完了範囲に含めない。
- 分類予測の数値部品は`../specs/weighted-class-probability-prediction/README.md`。3 tasks完了、全20条件をLunaが確認し最終統合GO。2026-10-04の全testsは2281 passed / 3 skipped、旧golden11ケース・最終golden3ケースを更新せず通過。二値/多クラスの確率化・混合・予測・平均有界損失と、重み状態へのテスト接続が完了範囲。新FedSDA全体run、モデルforward・学習は未移植。
- 全体・正解クラス別損失監視は`../specs/class-conditional-loss-monitoring/README.md`。3 tasks完了、全20条件をLunaが確認し最終統合GO。2026-10-04の全testsは2390 passed / 3 skipped、旧golden11ケース・最終golden3ケースを更新せず通過。単一e-SR・ClassESR混合・global候補位置・reset・明示損失接続が完成範囲。モデル統計からのbaseline推定、警報後操作、新全体runは後続範囲。
- 候補の将来損失評価は`../specs/post-alarm-candidate-loss-evaluation/README.md`。3 tasks完了、15条件をLunaが確認し最終統合GO。2026-10-04の全testsは2504 passed / 3 skipped、旧11/最終3goldenを更新せず通過。外部収集済みlossによる現行優先の既存適合選択・二分区間候補採否・診断値が完成範囲。収集session・model操作は未移植。旧採否/理由の丸め不整合はresearch.mdへ別修正候補として記録した。
- 保留標本位置FIFOは`../specs/pending-training-data-assignment/README.md`。3 tasks完了、14条件をLunaが確認し最終統合GO。2026-10-04の全testsは2580 passed / 3 skipped、旧11/最終3golden更新なし。位置の明示追加・超過解放・非破壊分割・全消費とpublic監視span接続だけが完成範囲。モデル帰属・payload・学習・警報後進行・終端方針は未移植。
- 旧実装の不具合・改善候補の正本は`../../docs/research/implementation-findings/README.md`。候補理由の丸め差、短い警報後の割当済み標本残留、終端FIFO末尾の未帰属を別記録で追跡する。今回の移植で修正しない。
- 警報後の候補/参照損失収集は`../specs/post-alarm-candidate-loss-collection/README.md`。3 tasksを実装し15条件の証拠を確認。2026-10-04の全testsは2664 passed /3 skipped、旧11/最終3golden更新なし。提案次位置からの固定参照loss系列、原子的拒否、規定件数到達とimmutable copy・既存採否への明示test接続が範囲。設定モジュールと宣言型だけへ依存許可を限定した。承認と最終統合判定の正本は同spec.json/review.md。
- 旧sessionが不足参照入力で部分更新する事実はLEGACY-004へ追跡記録を追加。正常clientへの影響は未確認で旧productionは変更しない。
- 有界損失の不変集計は`../specs/bounded-loss-moments/README.md`。一系列の件数・保存平均・偏差平方和、旧演算順の追加、n2以上の平均/標本分散と上位への明示test接続を実装。2026-10-04の全testsは2728 passed /3 skipped、旧11/最終3golden更新なし。依存境界・stdlib独立起動を確認、最終統合判定と承認の正本は同spec.json/review.md。モデル/class所属、batch seed、用途別baseline方針、学習・登録・統計merge、新全体runは後続。
- クラス統計の旧説明不整合と、不正class入力で旧全体統計が部分更新する事実をLEGACY-005/006へ記録。正常client/過去成果への影響は未確認。旧productionの修正と今回の数値移植を分ける。
- 用途別損失基準値は`../specs/loss-baseline-selection/README.md`。監視n1/clip、警報区間再利用n2/平均0除外、警報後履歴n2/平均0保持を、単一集計から選ぶpure方針として実装。上位への明示test接続、入力検査コピー・共有状態とexact上流型依存を検証。2026-10-04の全testsは2829 passed /3 skipped、旧11/最終3golden更新なし。全回帰と最終判定の正本は同spec.json/review.md。モデル/class統計map・seed/merge・snapshot時機・候補進行・新全体runは後続。
- モデル全体・クラス別損失統計は`../specs/model-and-class-loss-statistics/README.md`。明示seed/一括置換、独立snapshot、帰属損失の原子的更新を実装し、旧各更新への直接照合と監視/履歴参照への明示test接続を確認。2026-10-04の全testsは2885 passed /3 skipped、旧11/最終3golden更新なし。承認と最終判定の正本は同spec.json/review.md。モデル実体・class上限、batch seed算出、merge/ID変更/削除時機・学習・新全体runは後続。
- LEGACY-006には旧不正class入力による部分更新と新storeの原子的拒否の同入力テストを追記。旧production自体は未修正で、正常clientや過去成果への影響は未確認。
- batch損失からのモデル初期統計は`../specs/batch-loss-statistics-initialization/README.md`。外部計算済みloss/labelsから旧torch reduction順の全体/class集計を生成し、singleton非対称・class昇順/欠落と旧24ケースを照合、store/update/baselineへの明示test接続を確認。2026-10-04の全testsは2965 passed /3 skipped、旧11/最終3golden更新なし。承認/最終判定の正本は同spec.json/review.md。事前学習の逐次Welfordseed、model prepare/forward/学習/登録/送信、merge・新全体runは後続。
- 旧空batch登録でNaN初期統計を保持する事実をLEGACY-007に再現・同入力対照とともに追跡。正常client/過去成果への影響は未確認、旧productionは未修正。
- モデルID対応後の損失統計選択は`../specs/model-id-mapped-loss-statistics-selection/README.md`。一回ID対応/max全体件数・先着同数/欠落・zeroだけのserver補完、モデル/class順と旧18ケースの全field照合、独立コピーとstore/update/baselineへの明示test接続を検証。2026-10-04の全testsは3041 passed /3 skipped、旧11/最終3golden更新なし。モデル・学習データ・予測重み・現在の帰属IDの対応、登録/通信/server集計・新全体runは含めない。今回新たな旧正常不具合は未観測、既存LEGACY001–007は修正状態を変えない。承認/最終判定の正本は同spec.json/review.md。
- 次は候補開始・終了の進行、警報後のモデル帰属/client調整と、その前提となるモデル学習・統計mergeを依存順に仕様化する。未作成specの命名・実装を先取りしない。
- サーバ向け損失平均集約は`../specs/server-loss-mean-aggregation/README.md`。参加選別済みのclient順に統計件数で平均を加重し、M2zero/空・zero時Noneを両旧サーバ20ケースへ照合。上位whole server保持/置換→ID補完→store更新/baselineを明示test接続。2026-10-04の全testsは3103 passed /3 skipped、旧11/最終3golden更新なし。参加判定・モデルパラメータ集約/通信/状態保存・新全体runは後続。極大件数の範囲超過/除算失敗はLEGACY008へ再現と新拒否を記録、通常影響未確認/旧production未修正。承認・最終判定の正本は同spec.json/review.md。

SINE goldenの参照条件と所属はsingle-run-executionのreference-inventory.mdへ棚卸し済み。
今後は必要な追加契約を確認し、型付き設定と単一run基盤へ手法の処理部を接続する。
`configuration-foundation`の後続計画4〜8は未承認・未完了のまま保持する。

## 承認と検証

2026-10-03の追加指示により、今後は要求・設計・task・命名・実装について、gpt-6-lunaレビューと主担当の有用指摘の反映を承認として進める。レビューと採否・承認対象を記録し、毎段階で人間承認を再要求しない。
命名を変えた場合は承認を解除し、変更箇所をレビューする。
移植対象のgoldenと機能テストを各単位の検証へ接続する。
- 候補パラメータ初期化は`../specs/candidate-parameter-initialization/README.md`。全3tasksを完了しLuna最終統合GO。現在の学習先/評価済み最小loss/全保有等平均から独立snapshotを返す部品を実装し、全15dtype・順序・copyを旧helperへ直接照合した。test-onlyの公開候補評価mean接続とexact依存/CPU fresh smoke、全tests3182 passed/3 skippedを検証済み。承認・hash・9条件の証拠は対象specに記録した。LEGACY009へ極大float32平均の非有限化を記録、旧productionは未修正・通常影響未確認。モデル生成/適用/optimizer/学習/候補session進行/新全体runは後続範囲。
