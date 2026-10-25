# Claude交代後の研究作業と再開案内（2026-10-26）

現在地を判断する入口。各実験の条件・標準指標・テスト・独立レビューの証拠はリンク先の個別記録を正本とする。
[10月23日の引継ぎ](research-handoff-20261023.md)を受けた研究の続きであり、当時の「次にやること」を未実施と読み違えない。

## 1. 最初に読むものと現在の状態

1. worktreeの[AGENTS.md](../../AGENTS.md)、[共通引継ぎ手順](../../.kiro/steering/agent-handoff.md)、[resume](../../.kiro/steering/resume.md)。
2. 本書の2・3・5・6節。直近の根拠は[B/Wの交差解析](fedsda-s3-b-replay-transitions-20261026.md)。
3. 選んだ研究の個別記録と研究CLI README。全過去文書を一度に読む必要はない。

- 場所は`.worktrees/refactoring`、ブランチ`refactor/architecture`。mainへマージしない。開始時にHEADと差分を実測する。
- 本書作成の基点HEADは`db2b8c40`。これは文書・後処理の履歴で、個々の実験の数値生成commitとは区別する。
- **製品の既定は基準版12のまま。交代後のE1a/E2/S1/S3/B等は研究ツールによる検証で、製品へ採用していない。**
- 直近のB再現8runと後処理は完了。次のAIへ引き継ぐ実行中ジョブはこの作業からは無い。他者のサーバ利用状況は別途確認する。
- **次案E3は提案しただけで、設計・試作・実験は未着手。ユーザーはまだE3を選んでいない。今回のユーザー依頼は記録・引継ぎの確認。**
- 基準版13の変更、main統合、候補検証の処理削除、コミット保留資料の扱いは引き続き未決定。研究の続行許可をこれらの承認へ読み替えない。

## 2. 一連の作業の索引と結論

| 段階 | 行ったこと・得た判断 | 詳細の正本 |
|---|---|---|
| E1a | 過去の予測重みを混ぜる試作と予備対照。回復期の改善を支持せず、現行形は本規模化見送り | [E1a](fedsda-mixing-past-posteriors-e1a-20261024.md) |
| E2 | 予測クラス別の重み。予備対照とMNIST4・5000標本・4seedまで実施。専用枝単独は低下し、本実装見送り | [E2](fedsda-class-conditioned-prediction-e2-20261024.md) |
| S1成立性・安定性 | 影の学習者で符号長の負債を回収するかを非介入計測。交差はあるが再反転や誤警報相当でも発生 | [成立性](fedsda-shadow-learner-s1-diagnostic-20261024.md)、[安定性](fedsda-shadow-learner-s1-stability-20261024.md) |
| S1無変化/継続対照 | uraniumの120条件・240runを回収。単なる交差を安全な採用根拠にはできない | [設計](fedsda-shadow-learner-s1-controls-20261024.md)、[サーバ結果・標準指標](fedsda-shadow-learner-s1-controls-uranium-results-20261024.md) |
| optimizer・雑音 | optimizer継続/reset、5000標本への拡張、無雑音対照。optimizer履歴だけが主因という説明は支持せず | [対照](fedsda-shadow-adoption-and-optimizer-followup-20261024.md)、[拡張](fedsda-shadow-optimizer-expanded-20261024.md) |
| S1採用 | 本規模120条件・240run。SEA4概念2の平均改善なし、無変化条件でも低下。現行形は採らない | [サーバ結果](fedsda-shadow-adoption-uranium-results-20261024.md) |
| 共有部への再接続 | 採用時の接続による損失変化を計測。悪化はあるが全体低下の主因とは未確定 | [計測](fedsda-shadow-feature-reconnection-20261024.md)、[サーバ結果](fedsda-shadow-reconnection-uranium-results-20261024.md)、[現行との比較](fedsda-shadow-current-comparison-20261024.md) |
| 採用モデルの来歴 | 登録・統合・配布・局所予測使用を非介入追跡。未登録候補も局所予測には使われ、局所利得と全体差を区別 | [計測](fedsda-shadow-model-lifecycle-20261024.md)、[サーバ結果](fedsda-shadow-lifecycle-uranium-results-20261024.md) |
| 再接続後の新標本評価 | 新しい30標本で採用を保留・判定。即時採用を一貫して超えず、未採用 | [対照](fedsda-reconnected-candidate-holdout-20261024.md) |
| 単体優位と混合利得 | 同じ候補の現行モデル単体への優位と実際の混合への追加利得がずれることを確認 | [設計](fedsda-current-versus-mixture-benefit-design-20261024.md)、[結果](fedsda-current-mixture-benefit-results-20261025.md) |
| 候補除去・採用時情報 | 採用候補の局所的な有益/有害を調べ、4seedで採用時情報の識別を検証。安定したゲートの根拠不足 | [除去診断](fedsda-adopted-candidate-removal-results-20261025.md)、[4seedの識別](fedsda-adoption-information-seeds0-3-20261025.md) |
| S3出力層学習者 | 2500標本・4seedで試作。MNIST4で得失が大きく分かれ、他入力でも一般的な改善にならず | [S3](fedsda-output-layer-learner-s3-20261025.md) |
| S3寄与・二因子対照 | 候補単体の除去利得では全体差を説明できない。符号長評価と実候補学習を分離し、Bを次の候補とした | [寄与](fedsda-s3-contribution-diagnostic-20261025.md)、[二因子](fedsda-s3-factorial-20261025.md) |
| B未使用seed予備対照 | 手元2500標本・seeds4–7で改善。これは次の本規模実験へ進む根拠で、最終判断ではない | [予備対照](fedsda-s3-validation-20261025.md) |
| B本規模 | B240＋既定240、6データ×4間隔×10seed・5000標本。正負が割れ、汎用改善を支持せず | [準備](fedsda-s3-fullscale-20261025.md)、[到着結果](fedsda-s3-fullscale-results-20261025.md) |
| W20run | MNIST4 i100/500で初期値変更とhead学習変更を分離。初期値効果は小さく、学習範囲側に符号反転 | [準備](fedsda-s3-warm-control-20261025.md)、[到着結果](fedsda-s3-warm-control-results-20261025.md) |
| B8run再現 | seeds0/3/7/8×i100/500だけ計測付き再現。原B全非秒指標一致、保存Wと全入力一致。序盤と最終の符号が変わる | [準備](fedsda-s3-b-replay-20261025.md)、[到着結果](fedsda-s3-b-replay-results-20261026.md) |
| client×概念変化後経過 | 保存系列の40万対応観測を分割。損失は単一client・変化直後だけでなく、終盤利益には終盤の概念初出も寄与 | [交差解析](fedsda-s3-b-replay-transitions-20261026.md) |

上表は経緯の索引。各段階の「次にやること」は当時の提案で、後続で完了したものを再実行しない。
各記録に数値・条件・不足項目・レビューsession・検証範囲がある。全記録について同じ範囲の独立再計算が行われたという意味ではない。

## 3. 直近の解釈と取り違えやすい点

- 小規模で有望だったBは、本規模で条件依存が判明した。10seed平均と、事後に選んだ4seedの診断平均を混ぜない。
- Bの個別候補が局所的に有益でも、判定分岐・その後の学習・統合・予測が変わるため、方式全体の既定差と一致しない。
- 略号Bは文脈依存。以前の「候補除去診断B」と、S3二因子対照の「full-score-head-trainのB」は別物。
- 直近のDは製品既定、Wは現在モデル初期値で候補のadapter＋出力層を学習、BはWと同じ初期値で実候補を出力層限定学習する構成。
  Bの符号長評価はfull側のまま。S3は評価側もhead側へ変える。正確な定義は二因子対照の記録と`tools/research_s3_factorial/policy.py`を確認する。
- W−D/B−Wは条件付きのrun全体差。3構成だけで独立な加法効果や交互作用そのものを推定したとは言わない。
- **訂正済み事項**: D/Bとも系列がないという過去の説明は誤り。Dの元NPZは存在し、対象20件のW/D累積差を復元済み。
  Dに画像hashはない。WのhashをDへ転記せず、Dとの画像直接照合済みとは呼ばない。B系列は追加8runで回収済み。
- 真の概念、事後の候補除去利得、将来の正誤は診断用。採用時に利用できる情報とは区別する。
- 直近の部分群差はpooled差。以前の概念別seed等重み平均とは異なる。初期episodeは変化後窓に含めない。

## 4. データとツールの場所

`tools/`・研究記録はGitで共有済み。`results/`・監査スクリプトの一部・原系列はGit管理外。
**Git clone/pullだけでは全ての検算を再実行できない。** 保存先の移動後はREADMEに従い参照パスを明示する。

| 保存先（worktreeからの相対パス） | 内容・注意 |
|---|---|
| `results/fedsda-shadow-controls/20261024_uranium-s1-controls-seeds0-9-n5000` | S1対照の本規模 |
| `results/fedsda-shadow-adoption/` | 即時採用の本規模。具体的な日時・入力は対応する記録を参照 |
| `results/fedsda-shadow-reconnection/` | 共有部接続計測 |
| `results/fedsda-shadow-lifecycle/20261024_uranium-lifecycle-seeds0-9-n5000` | 来歴追跡 |
| `results/fedsda-s3-fullscale/` | ユーザー配置は直下。原D/B、metadata、baseline、runsを含む一式が必要 |
| `results/fedsda-s3-warm-control/20261025_uranium-mnist4-i100-500-seeds0-9-n5000` | W20件。predictionsも必要 |
| `results/fedsda-s3-b-replay/` | ユーザー配置は直下。B8件とpredictions、analysis_comparison、analysis_transitions、監査proof |
| `results/method-comparison-v12/20261023_220120_fedsda-v12-default-seeds0-9/fedsda/runs` | Dの元JSON/NPZ。コピーJSONのrecord_table_file_nameとの対応を検査して使う |

過去の小規模結果・診断出力の正確な日時ディレクトリは、表2の個別記録を参照する。
FedDrift/FedAvgの実装監査は`results/implementation-audits/feddrift-20261021/`と`fedavg-20261021/`にある。
これらの監査資料はユーザー希望によりGitへ入れない。

直近の後処理を再実行する場合（実験の再実行ではない）:

```powershell
../../venv/Scripts/python.exe -m tools.research_s3_b_replay.transitions results/fedsda-s3-b-replay --reference results/fedsda-s3-fullscale --warm-output results/fedsda-s3-warm-control/20261025_uranium-mnist4-i100-500-seeds0-9-n5000
```

入力一式があれば、原B再現一致・W/B入力/digest・参照hashを再検査してから分割する。
詳細は[研究CLI README](../../tools/research_s3_b_replay/README.md)。
直近の独立レビューPASS・研究checks3・関連/回帰pytest1403成功の証拠は交差解析の記録にある。
過去のテスト成功を、将来変更後の成功として流用しない。

## 5. 実行・報告の取り決め

正本は[共通引継ぎ手順](../../.kiro/steering/agent-handoff.md)と[tech](../../.kiro/steering/tech.md)。特に以下を読み落とさない。

- 研究ツール段階はspec不要。本実装すると決めたらcc-sddと独立CLIレビューを経る。既定へ黙って組み込まない。
- 手元の性能確認は2500標本以上・4seedを基本とする。本規模は5000標本・複数条件・多数seedをuraniumへ。
- uraniumは256コア、標準64workers。手元の14workers目安をサーバへ適用しない。各数値ライブラリ1thread。
- サーバの実行・転送はユーザー担当。ツールは検証・レビュー・pushしてからコマンドを提示する。小規模をサーバへ回すだけの提案は避ける。
- 同じ比較のseedで環境を分けない。Windows/WSL/uraniumのPython/NumPyが異なるため、厳密な再現は元環境で行う。
- 1時間以上の見込みなら55分おきに進行確認し、完了したら追加確認しない。独立した手元作業は並行できることを案内する。
- 報告では毎回、精度と既定差pp、上下通信量、計算/MAC、時間を示す。未保存はN/A、診断費用は分け、時間を負荷差のまま速度の優位としない。
- 次にできることと、その中のおすすめを毎回添える。試す構成を無制限に増やさず、診断と実性能の対照を切り離さない。
- 独立レビューは別CLIセッション。現在のモデル/effortと代替規則は共通引継ぎ手順を参照。内蔵子エージェントは使わない。

## 6. 次にできること（提案済み・未着手）

その後のユーザー指摘で、版13への更新も次の有力な選択肢として明示した。
[共有部固定の根拠と版12/13比較](fedsda-v12-v13-comparison-and-candidate-freezing-20261026.md)で、上限撤廃のみと両変更を区別し、主比較は全データ10seedに統一、MNIST4の20seedを別表にした。
次は版13の採用範囲を判断するか、新研究を選ぶか。共有部固定解除は合理性のない処置の削除とはせず、得失で決める。
以下の新研究案も提示済みだが、**版13も含め採用決定や実装指示はまだない。**

1. **推奨: E3、帰属の判定を予測重みの初期値へ反映**。
   保有モデルkへの再利用が決まった後、kを予測へ生かすまでの遅れを小さくする仮説。
   [IMPROVE-029のE3](improvement-candidates/improve-029-stronger-prediction-mixture-for-recurring-and-partial-concepts.md)に既存案がある。
   研究ツールで保有モデル再利用時に限定し、手元2500標本・4seed、SINE2/CIRCLE2/MNIST4の既定との対応比較と重み/損失診断を同時に行う案。
   帰属が誤った場合の悪化、FIFO再較正との証拠の重複を確認する。具体的な規則・条件数・ツール・コマンドは未設計。
2. **次点: Tの適応/複数Tの予測の混合**。
   [IMPROVE-016](improvement-candidates/improve-016-adapt-fixed-share-time-scale-and-pending-fifo-capacity.md)のT案。
   古い文書の「既定30」は当時の記述で、現在は共通T120。モデル別重みと同期/局所混合で別Tを勝手に増やさない。
3. **後回し: 検出を早めつつ誤警報を抑える**。
   [IMPROVE-027](improvement-candidates/improve-027-calibrated-sequential-change-detection-by-description-length.md)。
   設計・計算費用が大きく、過去の別基準の試作改善を現在の性能根拠にしない。
4. **並行候補: 現行FedSDAとFedDrift/FedAvgの比較条件・精度/資源差の整理**。
   実装監査には公式コードとの差がある。公式と同一の再現と呼ばず、既存監査を読んで比較の目的・条件を整理する。

E1b、E3、候補イベントと予測時刻の直接対応付けは未実施。
E1aの原因時系列診断にも未実施項目が残るが、優先して再開すると合意したわけではない。
以前からの未回収実験や保留資料はresumeの「未解消の事項」を参照し、一律に完了扱いしない。

## 7. 次のAIへ渡す文面

```text
.worktrees/refactoringで作業してください。AGENTS.md、.kiro/steering/resume.md、
.kiro/steering/agent-handoff.md、docs/research/research-handoff-20261026.mdを読み、
GitのHEAD・差分と必要なresults一式の有無を確認してください。
製品は基準版12のままで、E1a/E2/S1/S3/Bは未採用です。Bの本規模・W20・B8再現・
client×概念変化後経過時間の解析は完了しています。古い「次にやること」を再実行しないでください。
E3は次候補として提案しただけで未着手です。ユーザーの最新の依頼に合わせて再開し、
提案を実装許可や本実装の採用決定と取り違えないでください。
```

## 8. 引継ぎの検証

2026-10-26、主担当が本書の相対リンク41件、主要原データ8directory、Dの元NPZ240件、
本規模D/B480セル・W20・B8のcomplete件数、resumeの最新入口を確認した。
独立CLIはgpt-6-luna、effort medium、session `01a174ae-6c17-7eb0-9307-cd39d710643e`でPASS、指摘なし。
文書と状態の照合が中心のためLunaを選び、モデル/effortはCLI headerで確認した。
独立担当は入口から現在地・未着手・非承認の区別を復元し、同じリンク/存在/件数checkも実行した。
過去の全raw・全記録の再監査、NN/pytestの再実行はしていない。今回は文書のみ変更した。
証拠は`results/_sol_prompts/research-handoff-20261026-check.log`と`research-handoff-20261026-review.txt`。

