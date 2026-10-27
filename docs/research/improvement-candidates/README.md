# 具体的な改善候補

コードを調べて見つけた、範囲の限定された簡略化・効率化・アルゴリズム調整案を1候補1ファイルで記録する。旧実装・新実装のどちらも対象。各ファイルが詳細と状態の正本で、本書は入口と記録規約。

## 記録先の選び方

| 記録先 | 対象 |
| --- | --- |
| improvement-candidates/ | 複雑な処理の簡略化、計算量/通信量/メモリ削減、限定的な選択規則・設定の調整、その他具体的な改善案 |
| [implementation-findings/](../implementation-findings/README.md) | 不具合の疑い、契約・期待挙動との不整合、再現条件と影響の調査 |
| [研究バックログ](../research-backlog.md) | 広い研究方向、代替設計、アイデア段階の研究・実験候補 |

区別が難しい場合は、期待する契約との不整合が疑われるならimplementation-findings、改善余地という仮説なら本フォルダへ置く。未確認と明記し、同じ事象の詳細を二重に書かず相互リンクする。分類を変えた場合も元IDと移動先を記録する。既存のLEGACY記録は今回一括移動しない。

## 一覧

| ID | 件名 | 状態 | 記録 |
| --- | --- | --- | --- |
| IMPROVE-001（旧ALGO-001） | 警報区間の同率再利用候補で現行モデルを優先 | 保留（2026-10-13） | [詳細](improve-001-current-model-priority-on-reuse-ties.md) |
| IMPROVE-002 | 参照モデルの固定で不要なランダム初期化を省く | 見送り（2026-10-13） | [詳細](improve-002-skip-random-initialization-of-fixed-reference-models.md) |
| IMPROVE-003 | 固定した参照モデルの検証損失を到達時にまとめて評価 | 見送り（2026-10-13） | [詳細](improve-003-batch-fixed-reference-loss-evaluation.md) |
| IMPROVE-004 | 履歴基準が使えないモデルの警報区間評価を省く | 保留（2026-10-13） | [詳細](improve-004-skip-alarm-loss-evaluation-without-reuse-baseline.md) |
| IMPROVE-005 | 前区間の事前検査と吸収で損失評価を共用 | 対応済み（`baf072c`。2026-10-13に確認） | [詳細](improve-005-reuse-prevalidated-alarm-interval-losses.md) |
| IMPROVE-006 | 変化区間が不足した警報の後、検出の証拠を破棄しない | 採用（待つ案を、2026-10-14にFedSDAの既定にした。本規模の比較で、sine2の精度が+0.28〜+0.39ポイント、切替の遅れが約20標本短縮、ほかのデータでは精度がほぼ同じ。ADWINの2構成は即時のまま。[spec](../../../.kiro/specs/fedsda-default-waited-registration-and-alarm-wait/design.md)） | [詳細](improve-006-keep-detection-evidence-after-too-short-alarm.md) |
| IMPROVE-007 | 最終予測に不要な比較予測診断を選択実行する | 保留（2026-10-13） | [詳細](improve-007-optional-counterfactual-prediction-diagnostics.md) |
| IMPROVE-008 | 警報応答で、区間解決の引数の検査を区間の準備より前に行う | 保留（2026-10-13） | [詳細](improve-008-validate-alarm-resolution-inputs-before-preparation.md) |
| IMPROVE-009 | 依存境界testの方式を一本化し、機械生成の注入契約testを減らす | 見送り（2026-10-09ユーザー決定） | [詳細](improve-009-unify-dependency-boundary-tests.md) |
| IMPROVE-010 | 保留標本の並びの検査を1箇所にまとめる | 保留（2026-10-13） | [詳細](improve-010-share-pending-observation-validation.md) |
| IMPROVE-011 | 共同学習の反復が、参加したモデルを返す（計数の側の参加条件の重複をなくす） | 保留（2026-10-13） | [詳細](improve-011-report-participating-models-from-joint-training.md) |
| IMPROVE-012 | 観測標本の特徴を、配列で持つ（MNISTの784特徴で、標本ごとのメモリを減らす） | 保留（2026-10-13） | [詳細](improve-012-hold-observed-sample-features-as-array.md) |
| IMPROVE-013 | AdaHedgeの重みの証拠を、評価の層から移す（予測に使う構成ができたため） | 対応済み（証拠の部品と全標本のownerを手法の層へ、真の概念別のownerを評価の層へ分けた。値は変えない。2026-10-13） | [詳細](improve-013-move-adahedge-weight-evidence-out-of-evaluation-layer.md) |
| IMPROVE-014 | adapterを持たない構造も扱う分類器の、クラスと属性の名前を、構造に依らない名前へ改める | 対応済み（`ModelArchitectureClassifier`、属性`model_owned_feature_transform`。goldenは、パラメータ名のキーだけを書き換えた。値は変えない。2026-10-13） | [詳細](improve-014-rename-classifier-that-may-have-no-adapter.md) |
| IMPROVE-015 | Fixed-Shareの時間尺度を、帰属の保留（FIFO）の容量から分けて指定する | 採用（2026-10-16、[spec](../../../.kiro/specs/fedsda-fixed-share-time-scale-axis/tasks.md)） | [詳細](improve-015-separate-fixed-share-time-scale-from-pending-fifo-capacity.md) |
| IMPROVE-016 | Fixed-Shareの時間尺度Tと帰属の保留の容量を、clientがその場で決める | 未検証・未採用（2026-10-16） | [詳細](improve-016-adapt-fixed-share-time-scale-and-pending-fifo-capacity.md) |
| IMPROVE-017 | モデル集合変更時に既存モデル間の重み比を保つ | 未採用（R/Pの複数seed対照を完了。2026-10-19） | [詳細](improve-017-preserve-existing-weight-ratios-on-model-pool-changes.md) |
| IMPROVE-018 | 候補の共有部統合で既存モデルへの影響を制限する | 一部採用（候補学習中の共有部の固定を、候補検証なしと組み合わせて既定にした。2026-10-20、基準版7。候補検証ありとの組合せは不採用） | [詳細](improve-018-protect-existing-models-during-candidate-backbone-integration.md) |
| IMPROVE-019 | クラスの支持数と対損失差で候補を検証する | 未検証・未採用（2026-10-17） | [詳細](improve-019-candidate-validation-with-class-support-and-paired-losses.md) |
| IMPROVE-020 | 新しいモデルの登録時のクロス評価の通信を減らす | 採用（2026-10-20、配布とクロス評価の入れ替えを基準版8の既定として実装。精度は同じで下りが減る） | [詳細](improve-020-reduce-cross-evaluation-communication.md) |
| IMPROVE-021 | 保有モデルの再利用の判定を固定のしきい値から検定へ替える | 別の形で採用（2026-10-21、基準版10。警報の後の変化点と帰属先を最小記述長の原理で選ぶ。経緯: 2026-10-20再開。累積の対数損失で選び、新しいモデル側の学習者を初期化し直す案Aは、判定の正しさ69.6%→75.0%、手元の3 seedで精度は既定と同程度——−0.47〜+0.91pp——、下りは0.93〜1.12倍。本規模——5 seed——では精度差−0.63〜+0.43pp、MNIST2の全間隔で改善、下りは0.87〜1.21倍。検定の案Bは再利用へ偏る見積り） | [詳細](improve-021-statistical-test-for-held-model-reuse.md) |
| IMPROVE-022 | 候補の学習の早期終了を単純にする | 見送り（2026-10-20。固定エポックはMNISTで大きく下がり、SINE2では上がる） | [詳細](improve-022-simplify-candidate-early-stopping.md) |
| IMPROVE-023 | 候補検証を既定から外した後に残った設定を整理する | 進行中（2026-10-28、ユーザーの判断で案2: 候補検証の処理ごと消す。論文では候補検証に触れない。[spec](../../../.kiro/specs/remove-post-alarm-candidate-validation/requirements.md)、task 1完了・task 2実装中。経緯: 2026-10-20は4案の4番目でユーザー判断待ち） | [詳細](improve-023-remove-settings-left-from-candidate-validation.md) |
| IMPROVE-024 | 警報の時点で、推定した変化点より後に解放された標本を、古いモデルの学習データから外す | 見送り（2026-10-21ユーザー決定。経緯: 2026-10-20に保留。2変種とも、手元の3 seedでも本規模——2026-10-21、5 seed——でも一般的な改善にならない。本規模の既定との差は、外すだけ−0.59〜+0.26pp、区間へ足す−0.46〜+0.22pp） | [詳細](improve-024-withdraw-released-samples-after-estimated-change-point.md) |
| IMPROVE-025 | サーバのクラスタリングの判定を、損失の距離へ単純にする（閾値の置き方を見直す） | 未採用を提案（2026-10-21。版10の10 seedで、CIRCLE2の下りは0.80〜0.91倍に減るが、MNISTで精度が下がる——最大−0.49pp——。閾値を固定しない(a)検定・(b)分布の境目は、対ごとの見積りで今の判定より悪い） | [詳細](improve-025-simplify-clustering-decision-to-loss-distance.md) |
| IMPROVE-026 | サーバのクラスタリングの判定も、符号長の比較で行う | 保留を提案（2026-10-21。符号長で統合を決める3つの形と、統合で代表のパラメータを保つ形との組み合わせを手元で試した。代表を保つと、新しく学ぶ費用の判定のMNISTの崩れは止まるが、既定より良くならない。当面は閾値0.1のまま） | [詳細](improve-026-server-clustering-by-description-length.md) |
| IMPROVE-027 | 警報を出す時点も、誤警報の率を抑えた符号長の比較で決める | 未検証・未採用（2026-10-21。毎標本の比較はSINE2で+0.3〜+0.8ppだが誤警報で崩れた。尤度比のe-過程で誤警報を抑える案。研究段階） | [詳細](improve-027-calibrated-sequential-change-detection-by-description-length.md) |
| IMPROVE-028 | モデルの統合を、登録の直後の1回の判定から、証拠を積み上げる仕組みへ替える（A: clientの利用の実績、B: 連続的な共有、C: 作ったときの証拠の引継ぎ、D: 種類に依らない概念の表現） | 一部採用（評価のclientを絞る変更は基準版12、評価のclientの上限の撤廃は基準版13——2026-10-27——として採用。候補の共有部の固定は維持し、外す枝は基準版13のablationで測る。拒否権は既定にしない。案B1（パラメータの共有）・案C（蒸留）は採らない。案Eは未着手） | [詳細](improve-028-evidence-accumulating-model-consolidation.md) |
| IMPROVE-029 | 予測の混合を概念の再来・部分変化に強くする | E1a/E2は未採用。E3は効く余地の見積りが上限で0.075pp以下のため試作しない（2026-10-26、主担当） | [詳細](improve-029-stronger-prediction-mixture-for-recurring-and-partial-concepts.md) |
| IMPROVE-030 | client側で部分変化を新モデルへ分ける | S1/S3/B本規模・W/B再現・交差解析まで完了、未採用。2026-10-27: SEAの見逃しを検討。概念を分ける価値の上限はSEA2で2.9pp・SEA4で4.9ppだが、基準版13はほとんど取れていない。今の検出は帯の狭い変化に区間の長さ以上の標本を要する。oracleの天井は+1.1〜1.8ppだが、警報だけの正解（5・50標本目、保留の容量200も）は効かず、ボトルネックはモデルを分ける段と分かった。塊単位の事後の再帰属（製品外）は条件がそろえば分けられるが、製品化の費用に見合わないため、ユーザーの判断でSEAの検討を止めた（2026-10-27）。事後の再帰属は将来の候補 | [詳細](improve-030-split-partially-changed-concept-on-client.md) |
| IMPROVE-031 | 損失の監視の基準を、損失の低下の検出で再設定する | 採用（2026-10-27、基準版13。評価clientの上限の撤廃と合わせて既定にした。本規模で採否の基準を満たす） | [詳細](improve-031-loss-monitor-rebaseline-on-loss-decrease.md) |

## 決定（2026-10-13、ユーザー）

上の表の状態と、各記録の「判断（2026-10-13）」が正本（Claude Codeの提案をユーザーが了承）。保留は、効果や頻度を測ってから採否を決める（測るまで実装しない）。経緯は[実装findingの決定](../implementation-findings/README.md#決定2026-10-13ユーザー)。

## 引継ぎ時の判断案（2026-10-13、Codex。上の決定より前の案）

以下は**未検証の主担当推奨案であり、採用決定ではない**。ユーザーはClaudeにも再検討させて最終判断することを希望している。まず[実装findingの判断案](../implementation-findings/README.md#引継ぎ時の判断案2026-10-13codex上の決定より前の案)の正常経路の整合性を検討し、効率化・構造整理・研究アルゴリズム調整を同じ修正へ混ぜない。

| 対象 | Codexの判断案 | 判断に必要な条件 |
|---|---|---|
| IMPROVE-001/006 | 研究上の選択規則・検出の変更として保留を推奨 | 頻度、回復、誤警報、精度の比較と採否基準が要る。006はLEGACY-002の重複帰属防止と関係するが、証拠resetの変更をFIFO修正に混ぜない |
| IMPROVE-002/003 | 数値/RNG変更を伴う効率化として、効果を測ってから採否判断を推奨 | 002は乱数消費、003はバッチ丸めと採否が変わりうる。不要に見える処理を消すだけでは同値の変更にならない |
| IMPROVE-004/005/007 | 効率化の優先検討候補。採用は効果と契約の検証後 | 004は入力拒否と候補選択、005は状態更新前の検査とモデル世代、007は予測に使うAdaHedgeとの区別と未計測診断の保存を維持する。処理省略は計算量・記録を変えうる |
| IMPROVE-008 | 正常値維持と拒否時の状態不変を確認できれば採用を推奨 | 契約外入力での例外時点を意図的に変更する。既存testが固定する準備後の部分更新と、短い警報で検査を追加する点を明示する |
| IMPROVE-010/011 | 現行コードで重複が残るか再確認して、整理の利益があれば採用を推奨 | ownerや戻り値を増やすことで複雑化しないこと。011でユーザーが許容したNEW-003を無断変更しない |
| IMPROVE-012 | MNISTのメモリ負荷が実験上の制約なら優先する条件付き候補 | 等値・不変性・所有権・並列保存/再開の契約と、実測の時間・メモリを確認する。合成データ中心なら優先度を下げる |
| IMPROVE-013/014 | 手法層への配置と構造に合う命名の整理として採用を推奨 | 013は真の概念別の比較用証拠を手法の判断へ流さない。014はindependent構成も含め役割に合う名を検討する。いずれも実データの列名・state_dictキー・golden hashへの波及を確認する |
| IMPROVE-009 | ユーザー決定の見送りを維持 | 依存境界の既存testを一括削除・方式変更しない |

候補の効果は未測定で、採用順も未決定。挙動を維持できた整理では基準版1をそのまま使う。意図した数値・選択・診断の変更は、変更点ごとの期待差と比較結果を確認してから新基準版へ反映する。先に期待値を再生成して一致させることを採用根拠にしない。

## 記録・更新の規約

IDはIMPROVE-連番、ファイル名は`improve-<番号>-<対象と案>.md`。次はIMPROVE-032。候補の発見時点で記録し、実験済みである必要はない。

各ファイルへ以下を記載する。未確認・未定の項目はそのまま明記する。

- 日付、対象コード/構成、基準commit、発見したspec。
- 種別（簡略化・計算効率・通信効率・メモリ・アルゴリズム/設定調整等）と状態。
- コードで確認した事実・具体的な箇所。効果の仮説とは分ける。
- 変更案、期待する利点、悪化やトレードオフの懸念。
- 挙動を維持する整理か、数値/選択/RNG/診断が変わる可能性のある変更か。
- 必要な比較・指標・採否基準、関連する不具合/研究案/spec。
- 検証結果、採用/保留/見送りの理由、変更commit。結果が否定的でも削除しない。

状態は「未検証・未採用」「検証中」「保留」「採用」「見送り」。候補の記録は変更・golden更新・長時間実験の承認ではない。旧挙動を維持する移植へ混ぜず、採用時は変更の影響範囲に応じた別spec/検証で扱う。
