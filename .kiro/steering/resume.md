# リファクタリング再開案内

2026-10-29更新。エージェント間の引継ぎ用。承認・進捗・検証証拠の正本は各spec。この文書は現在地への入口とする。

## 現在地

- 2026-10-29: [fedsda-reverse-prequential-change-interval-search](../specs/fedsda-reverse-prequential-change-interval-search/tasks.md)はTask 1〜6を完了。既定は変えず、基準版15を維持する。本規模の640 runで既定との精度差は平均+0.01〜+0.02pp（[IMPROVE-032](../../docs/research/improvement-candidates/improve-032-reverse-order-prequential-code-for-all-change-intervals.md)）。採否は未判断（ユーザーが判断する）。
- **最新の総括・再開順は[研究の引継ぎ（2026-10-26）](../../docs/research/research-handoff-20261026.md)**。Claude交代後の一連の研究、結果の所在、訂正、未着手事項をまとめている。
- **候補検証の削除は完了、現在の基準版は15**（[対象spec](../specs/remove-post-alarm-candidate-validation/tasks.md)、2026-10-28）。版15のgoldenは生成元`c13149ea`（WindowsとWSL）。版1〜14の保存済みgoldenと乱数のfixtureは維持する。版14で始めた掃引は、版15のコードでは再開できない（始めたときのコードで再開する）。
- 実装レビュー・全pytest・WSLの結果の正本は対象specのImplementation Notes。E1a/E2/S1/S3/B・E3とmain統合は未採用／未決定のまま。v13のablation（共有部の固定を外す枝を含む）と、評価の設計（研究バックログの9）は未着手。

- 2026-10-27: [IMPROVE-031](../../docs/research/improvement-candidates/improve-031-loss-monitor-rebaseline-on-loss-decrease.md)は上限撤廃とともに既定へ実装済み。基準版13として既定にした（上記spec）。
- 2026-10-27: 評価の設計（流れの長さ、期間別の精度、FedDriftとの公正な比べ方）は未決。ユーザーが後で考える。検討の材料は[研究バックログの9](../../docs/research/research-backlog.md)。

## 研究の経緯（過去時点の推奨を含む）

以下は各時点の記録。「次」「未実施」だけを拾って再開せず、最新総括と後続記録で現在の状態を確認する。

- B/Wのclient×概念変化後経過標本数の解析（2026-10-26）: [結果](../../docs/research/fedsda-s3-b-replay-transitions-20261026.md)。追加NNなし。初期/初出/再訪・排他的経過窓で全40万観測を分割し、純増正解i100−422/i500+283へ復元。i100seed7は9client負、概念1の損失は再訪が大きい。i500概念3は500標本以降にも純減240が残り、変化直後だけの問題ではない。i500seed7終盤の+218正解のうち+146は終盤に概念1へ初出した2clientのepisodeで、長く待った効果と区別する。機構の因果特定・運用上の安全な採用規則は得られず本実装保留。おすすめはB単独の追加NN実験をいったん区切って別仮説へ予算を移すこと。研究checks3件・Luna medium独立レビューPASS。検証の正本は結果文書。

- B計測付き再現の到着結果（2026-10-26）: [結果](../../docs/research/fedsda-s3-b-replay-results-20261026.md)。ユーザー配置はresults/fedsda-s3-b-replay直下。B8run成功・失敗0、原Bの全非秒指標/イベント一致、保存Wとの入力8対・全80万予測を検算。選定4seedのB−Wはi100−0.2110pp/i500+0.1415pp。i100はseed7が純減422中411正解分を占める。seed7はi100で序盤正→最終負、i500で序盤負→終盤正となり、序盤の符号だけで安全に選べない。i500概念1は4seed全て正だが概念3の悪化と併存。本実装保留。client×概念変化後経過標本数の交差解析は上記の後続記録で完了。Luna medium独立レビューPASS。

- Bだけの計測付き再現（2026-10-25準備、2026-10-26結果到着）: [記録](../../docs/research/fedsda-s3-b-replay-20261025.md)、[uraniumコマンド](../../tools/research_s3_b_replay/README.md)。MNIST4 n5000・seed0/3/7/8・i100/500のB8runのみ追加、保存済みW8と既定D8を再利用。原Bの秒以外の全指標/イベントと入力一致が必須。累積/区間/概念別救済悪化・全標準指標を出力。Luna medium独立PASS。手元WSLは原B/WとPython/NumPyが異なるため、同じuranium環境で実行し結果受領済み。結果は上記の後続記録。

- 限定W結果（2026-10-25、解析・独立レビュー完了）: [結果](../../docs/research/fedsda-s3-warm-control-results-20261025.md)。W20＋D/B40、失敗0。W−Dはi100/+0.0236pp、i500/+0.0238pp。B−Wは−0.1504/+0.1220ppで、このD→W→B分解では主に学習範囲側に間隔依存が出た。W全100万予測・B画像署名20対一致、i100/500も全10seedで同じ入力。初期値変更なし10runはD/W正誤系列と全標準値が完全一致。前回のD/B系列なしという説明を訂正：Dの元NPZ240件があり、対象20件からW/Dのlabel/concept対応・累積差・概念別精度を追加学習なしで復元。D画像hashは無く直接画像照合済みではない。概念1のB−Wが−0.4159/+0.9582ppと逆転。Luna監査PASS、本実装見送り維持。次に追うならBだけ4seed(0/3/7/8)×2間隔の8計測再現で既存W系列と比較（到着解析済み、上記参照。診断用で性能検証ではない）。

- 限定W対照と累積計測（2026-10-25、実験完了）: [記録](../../docs/research/fedsda-s3-warm-control-20261025.md)、[コマンド](../../tools/research_s3_warm_control/README.md)。MNIST4 i100/500・seeds0–9・n5000、Wだけ20runをuraniumで追加し、D/B40件を再利用。W/Bの画像/ラベル/概念の全入力署名を直接照合、予測系列も追加NN0で保存し、進行/再開対応。専用32＋二因子27の59件、回帰29件成功、Luna独立PASS。新レコーダの4概念/10クラス混同を実runで検出・修正し、W2500 seed3の全非秒指標/イベント・25000予測の入力/digestが旧参照と完全一致。既存B/D2500系列の累積差も検算済みで5000の原因とは呼ばない。W本規模結果は上記の後続記録、本実装未決定。

- B本規模結果（2026-10-25、検算・独立レビュー完了）: [結果](../../docs/research/fedsda-s3-fullscale-results-20261025.md)。ユーザー配置は`results/fedsda-s3-fullscale/`直下。B240＋既定240、失敗0・全480checkpoint/原記録/source一致。24条件の平均差は正12/負12、等重み平均+0.00875pp（記述値）。MNIST4 i500+0.1458pp・7/10正、i100−0.1268pp・4/10正、全4間隔平均+0.00395pp。B発動なし78対は精度/上下通信/全MAC/期間精度が完全一致、Luna結果レビューPASS。予備実験より汎用改善の根拠は弱まったので本実装/既定化は推奨しない。当時の次のおすすめだったW20runと累積差計測は、上記の後続記録で完了。

- B本規模CLI（2026-10-25、実験完了）: [計画/検証](../../docs/research/fedsda-s3-fullscale-20261025.md)、[コマンド](../../tools/research_s3_fullscale/README.md)。ユーザー指摘で960run案を縮小し、W/S3を外す。同条件の既定v12はclean commitと現製品srcが同じで全設定/環境を検証して240件再利用、新規Bだけ240run（6データ×4間隔×10seed、n5000、uranium64workers）。`NEW n/240`表示、checkpointと--resume対応。環境等が不一致なら流用せず停止、--baselineなしの場合だけ既定+Bの480run。旧既定の秒/準備MAC/概念精度/画像hashはN/Aで、画像列の直接照合済みとはしない。専用25＋既存47の72件・回帰29件成功、Luna独立PASS、実記録dry planで新規240確認。本規模結果は上記の後続記録、製品変更なし。

- B未使用seed確認とseed3追跡（2026-10-25、手元の未使用seed16run完了）: [研究記録](../../docs/research/fedsda-s3-validation-20261025.md)。未使用seeds4–7・MNIST4・i500・n2500、D/W/B/S3の16runを手元8workersで完了。ユーザー指示で小規模実験は手元、本規模はuraniumと配置を訂正。主評価B−Dなど規則は事前固定。Bは既定比平均+0.513pp（SD0.267）、4/4seedで正、W比+0.362/S3比+0.514ppも全4seedで正。上り+0.228%、下り+0.602%、MAC+0.893%。ソースhash・入力一致・標準指標の独立検算成功、Luna結果レビューPASS。手元seed3 Bは原参照と秒以外全指標/イベント一致、10万予測を検算。Wと71で同じ判断/初期値、候補学習後だけ異なり72で予測分岐、Wだけ109/Bだけ126で判断。既定比−159正解は主に概念1/安定期。head12件の局所寄与は11正/1件−1、同時除去+760で全体差とは別。55件＋別の既存回帰29件成功、Luna最終PASS。共用解析器の片側判断見落としだけ修正、製品/旧数値実行部は未変更。おすすめは次にuraniumで5000標本・複数データ/間隔・多数seedの本規模準備。本規模用の独立CLIは上記の後続で準備済み。ゲートの事後調整・本実装はしない。

- S3二因子対照（2026-10-25、完了）: [研究記録](../../docs/research/fedsda-s3-factorial-20261025.md)。MNIST4・間隔500・4seed・2500標本、新規A/B 8run＋既定/W/S3 12run再利用。B（評価full・実候補head）が既定比平均+0.268pp、3/4seedで正だがseed3−0.636pp、MAC+1.754%。S3は−0.006pp。head評価を加えるB→S3は平均−0.274ppで、評価/学習の条件付き効果と交互作用を分離した。全入力署名・20cellの標準指標・差の恒等式を検算、専用27件＋既存29件成功、Luna実装/結果レビューPASS。おすすめはBを固定した未使用4seed以上での独立比較。並行してseed3の非介入追跡が可能。製品・既存研究ツール未変更、本実装は未決定。

- S3寄与診断（2026-10-25、完了）: [診断記録](../../docs/research/fedsda-s3-contribution-diagnostic-20261025.md)。MNIST4・4seed・2500標本・3構成12run、元参照12対と秒以外の全保存値が一致。全30万予測の被覆/入力対応と503 MDL判断を検算。S3候補10件は統合前の単体除去で全件有益、悪化したseed3でも同時除去すると正解−145。全体の既定差とは別で、有害S3候補の選別だけでは説明できない。全seedで同一状態からの判断分岐を確認、seed2は最初のS3採用前に既定比正解−119。次は同じln2/初期値で符号長評価だけ・実候補学習だけを変える2因子対照へ進んだ（結果は上記の後続記録）。製品・既存ツール未変更、14件＋既存29件成功、Luna最終PASS。初回の被覆指摘と修正・独立NumPy検算も研究記録へ保存。

- S3出力層限定学習の独立対照（2026-10-25）: [研究記録](../../docs/research/fedsda-output-layer-learner-s3-20261025.md)。4seed・2500標本、SEA4通常/固定2・SINE2通常i100、MNIST4 i500、3構成48run相当（新規36＋既定12再利用）。S3の既定差はSEA4通常+0.021pp（概念2は−0.0133pp）、固定2は0、SINE2−0.173pp、MNIST4−0.006pp。合成3入力でS3候補0、MNIST4は10採用だがseed0/+1.084、seed3/−0.888ppと不安定。本実装・3学習者化・本規模実験は推奨せず、次候補は案②の特徴に応じた混合の成立性診断。製品src・既存研究ツール未変更。専用9件・既存29件成功、Lunaの結果・最終実装レビューPASS。途中の全体収集失敗とレビュー証拠の補完も研究記録へ残した。

- 採用時情報の4seed識別（2026-10-25）: [固定判定と未使用seedの評価](../../docs/research/fedsda-adoption-information-seeds0-3-20261025.md)。既存seed0/1を再利用し、2/3の2500標本24 runだけ追加、計4seed24条件48run・41候補。16個の採用時観測量だけを使い、0/1で単一特徴/閾値を固定、新結果未到着を時刻/hashで記録。新2/3の損失ラベルBA75.71%（有益7/7、有害6/10）、seed別87.5/50。4fold LOSOでは選択特徴が変わりBA50〜58%、平均54.05。正誤ラベルはtest非中立7件だけ、採用後30標本を揃えた損失testBA52.09%。安定したゲートの根拠は不足、追加の閾値探索をいったん終え別仮説へ移る推奨。新seedの独立参照照合は未実施（0対）、同一Bソースと保存確率/target/条件を検算済み。専用3件・回帰29件成功、Luna実装PASS。製品・既存研究ツール未変更、新ゲート未実装。

- 実採用候補の除去診断B（2026-10-25）: [2500標本24 runの結果](../../docs/research/fedsda-adopted-candidate-removal-results-20261025.md)。6入力・間隔100・seed0/1、元S1即時採用の全12対と秒以外の全保存値・MAC・採用経路・12トレース38867行一致。253ソースhash一致、追加forward0。24採用すべてで統合前を観測、12108延べ対（unique11740）、undefined0。候補ありでSEA4通常は正解化34/不正解化6、固定2は21/77（両seed不利益）、SINE2通常は187/4（全てseed0）。寄与が無視できるという結果ではなく、有益/有害を分ける余地がある。ただし局所除去差と既定からの全体精度差は別物。専用6件・既存29件成功、Luna実装PASS。次に行うなら保存記録から採用時に有益/有害を識別できる情報があるかを調べることを推奨、独立seed検証前に閾値や本実装を決めない。追加実験・新ゲートは未着手、製品変更なし。

- 現行単体差と混合追加利得の対照A（2026-10-25）: [2500標本24 runの結果](../../docs/research/fedsda-current-mixture-benefit-results-20261025.md)。6入力・間隔100・seed0/1、元wait30と全12対で秒以外の全保存値・MAC・採用経路・12トレース41781行一致、追加forward0。28保留→22完了/6警報中止、完了660標本・中止59標本。主alpha0.1で現行差と混合追加利得が7/22窓で不一致（現行+/追加非正2、逆5）。正誤は正解化1/不正解化0で、全体精度や新ゲートの利得とは呼ばない。seed等重み/候補平均/標本平均・厳密0は別解析モジュールで保存。専用9件・既存検査29件成功、Luna修正後実装PASS。推奨次候補はBの実採用後・統合前の除去診断を手元2500標本で実装・検証すること。Bは未実装、製品変更なし。

- 来歴追跡のuranium結果（2026-10-24）: [120条件・240 runの検算](../../docs/research/fedsda-shadow-lifecycle-uranium-results-20261024.md)。全250ソース、計測なし採用との全保存値・MAC・採用経路（秒除外）、120トレース654561行一致。488採用中366登録/122終了時未登録保留、未登録122件もすべて局所予測で使用。359統合中339件は登録と同じ同期、13件は最終同期。統合後代表の係数を元候補の寄与率と解釈しない。[単体差と混合利得の対照設計](../../docs/research/fedsda-current-versus-mixture-benefit-design-20261024.md)を追加。推奨Aは同一wait30候補・同じ新標本で現行単体差/実混合との差/仮想追加利得を非介入計測。補助Bは実採用後の統合前だけ候補除去差を測る。A/Bは未実装・未実験。次はAを手元2500標本で検証する候補。製品変更なし。

- 再接続後の新標本での採用保留（2026-10-24）: [設計と2500標本の4構成対照](../../docs/research/fedsda-reconnected-candidate-holdout-20261024.md)。6入力・間隔100・seed0/1、12条件48 runが全成功。概念固有部固定・現行共有部追随で新30標本を評価しS>0なら採用。gateは29保留・14採用・11棄却・4警報中止。精度は既定に対して2条件改善/1同値/3低下、即時採用を一貫して超えず、本実装は推奨しない。専用11件・既存検査29件成功、Luna実装レビューPASS。単体優位と混合利得の切分けは上記の対照設計へ進んだ。既存即時採用のuranium来歴追跡は別経路で、上記の通り回収・検算済み。製品・既存研究ツールは未変更。

- 接続後の影と現行モデルの対照（2026-10-24）: [三つの固定コピーの比較](../../docs/research/fedsda-shadow-current-comparison-20261024.md)を追加。600標本12 run・1500標本/seed0〜1の24 runが成功。smoke全6対で元採用方式と秒以外一致。pilot31採用のうち直近で接続後の影が良い27件にも、将来で悪い12件が残る。直近窓だけのゲートは推奨せず、現在共有部への接続後に新しい標本で証拠を取る対照が次の設計候補（未実装）。uranium本規模来歴追跡の64workersコマンドは提示済み、サーバ開始は未確認。製品・既存研究ツールは変更なし。

- 共有部接続のuranium結果（2026-10-24）: [本規模の検算と判断](../../docs/research/fedsda-shadow-reconnection-uranium-results-20261024.md)。120条件・240 run成功、全250ソース一致、計測なし採用方式と全120対で秒以外の全保存値・MAC・採用記録・全654561トレース行一致。488採用中、直近348悪化/2同値/138改善、将来282/2/204。再接続による損失悪化は確認できたが、全体精度低下の主要因は未確定。推奨は接続後の影と現行モデルを同じ過去標本で比較する手元対照の準備。本規模来歴追跡は別出力で並行できる。今回追加実験なし・製品変更なし。

- 採用モデルの追跡（2026-10-24）: [登録・送信・統合・配布・予測係数の非介入計測](../../docs/research/fedsda-shadow-model-lifecycle-20261024.md)を追加。6入力600標本の12 run成功、元採用方式と全6対で秒以外の全保存指標・MAC・採用経路一致。9採用中5件は終了時も未登録保留、4件は登録と同じ同期で統合（3件は最終同期）。小規模の頻度を本規模へ一般化しない。共有部接続のuranium実験はユーザーが開始済み。回収後に両経路を分けて評価する。製品変更なし。

- 共有部接続の計測（2026-10-24）: [非介入の計測](../../docs/research/fedsda-shadow-feature-reconnection-20261024.md)を追加し、専用3件成功、Luna実装レビューPASS。6入力600標本の12 runが成功し、以前の採用方式と全6対で秒以外の保存値・差引後MAC・採用経路一致。9採用すべてに共有部差、直近損失は6悪化/3改善。主要因とは未確定で、次はuraniumの同じ120条件で計測する推奨。参照照合は同じ環境の結果を使用。製品変更なし。

- ②のuranium結果（2026-10-24）: [採用120条件・240 runの結果](../../docs/research/fedsda-shadow-adoption-uranium-results-20261024.md)を回収・検算。失敗0、249ソース一致、全baselineが旧サーバと秒以外一致。採用488件。SEA4全体差+0.0164/−0.0194pp、目標の概念2は−0.3203/−0.0340pp。SINE2通常は+0.0448/+0.1418ppだがseed差と追加費用あり。無変化8条件は平均低下。現行形の本実装は推奨せず、採用時の共有部再接続と採用モデルの登録・統合経路の小規模計測が次の推奨。製品変更なし。

- ①の拡張と雑音対照（2026-10-24）: [5,000標本・4 seedの24 runと雑音なし8 run](../../docs/research/fedsda-shadow-optimizer-expanded-20261024.md)を手元で完了、失敗0。各データ条件内で精度差0pp・通信不変。概念2の同時D/C交差はoptimizer継続/resetとも3/24で主因を支持せず、全体39件の同時交差は全て再反転。無雑音・無変化でも交差が残る。単純なC条件の追加採用は見送る。次の推奨はユーザーがuraniumで開始した②の実際の精度・費用の確認。製品srcは変更なし。

- S1の次段階（2026-10-24）: ユーザー指示により[採用実験②とoptimizer対照①](../../docs/research/fedsda-shadow-adoption-and-optimizer-followup-20261024.md)を並行する。②はuranium、①は手元。製品srcは変更しない。②のツールはpush・コマンド提示済み（サーバ実行はユーザー担当）。①は手元の12条件・24 runを完了、精度差0pp・通信不変。D/C同時交差はoptimizer継続17件、リセット16件で主因とは断定できず、概念2の影は1件だけ。次の推奨は②の本規模の精度差・費用の確認。

- 標準指標の一覧化（2026-10-24）: S1対照の120条件から、精度・上下通信・MAC・経過時間と診断を併記する[一覧と方法](../../docs/research/fedsda-shadow-learner-s1-controls-uranium-results-20261024.md)を追加。analysis_standard_metrics.md、standard_metrics_by_seed.csv、summary_standard_metrics.jsonを元結果のディレクトリへ保存。今後の研究報告は毎回標準指標を明示し、次の候補と推奨も添える。次の推奨は引き続き初期化・optimizer履歴の対照設計。再実験はしていない。

- uraniumのS1対照の結果確認（2026-10-24）: [120条件・240 runの分析](../../docs/research/fedsda-shadow-learner-s1-controls-uranium-results-20261024.md)を記録。失敗0、全120対で既定と保存値一致・追加計算を除く計数一致。概念2でD/C同時交差は間隔100が13件、500が16件、各6 seedだが無変化でも発生し、D交差後の再反転も多い。D>0だけで採用する案は保留。推奨の次段階は初期化・optimizer履歴の小規模な対照設計。元結果はresults/fedsda-shadow-controls/20261024_uranium-s1-controls-seeds0-9-n5000。以後の報告には次の候補と推奨を添える。

- サーバ結果待ちの追加作業（2026-10-24）: [S1の逆転後の安定性と分析方針](../../docs/research/fedsda-shadow-learner-s1-stability-20261024.md)を作成。既存8 runを再学習せず分析し、概念2の8交差中6件が概念維持中に非正へ戻る一方、同概念の最終観測は8件とも正と確認。seed0件も含む集計・打切り・D/Cの安定性を調べる追加分析CLIを準備。サーバの途中結果は未参照。実験を開始する既存のCLI・診断は変更していない。

- 実験環境の最新訂正（2026-10-24）: uraniumは256コア・標準64 workers。手元16コア向けの14/各7をサーバの上限としない。CLIの案内と古い記述の所在は[tech.md](tech.md#研究実験の実行環境とコマンド2026-10-24更新)。今回のS1対照も`--workers 64`で案内する。

- 2026-10-24の次段階: [S1の無変化・オンライン継続対照](../../docs/research/fedsda-shadow-learner-s1-controls-20261024.md)を準備。6入力・seed1・600標本の対照12 runが成功、関連60件成功、Luna medium実装レビューPASS。製品変更・採用は無し。本規模は6入力×間隔100/500×10 seedの120条件（既定と診断で240 run）。uraniumでの転送・実行はユーザー担当、コマンドは研究記録。ユーザーが研究ツールのpushを許可。uraniumはrefactor/architectureをpullして専用shellで実行する（ZIPは予備）。本規模は結果回収・照合・分析完了。最新の判断は上記のサーバ結果の記録。

- 続く2026-10-24の研究: ユーザー指示で[S1の成立性診断](../../docs/research/fedsda-shadow-learner-s1-diagnostic-20261024.md)を実施。影を予測・採用に使わず損失差だけ追うツールを作成し、境界修正・Luna再レビュー後の5,000標本・4 seed・8 runは失敗0で完了（20261024_s1-feasibility-fixed-n5000）。SEA4概念2は24影中8件が概念変化前に初期差を回収、うち6件はseed1。他概念でも交差し、採用機能は未試作・未採用。次の研究候補は無変化対照・同じ学習手順の対照の準備。共通保存項目と追加計算を除く計数は既定と全8対一致。専用全体対照2件＋境界修正後関連55件成功、Luna medium PASS。研究ツールなのでspecは作らず、製品srcは変更していない。

- 2026-10-24の研究作業: [E1a](../../docs/research/fedsda-mixing-past-posteriors-e1a-20261024.md)と[E2](../../docs/research/fedsda-class-conditioned-prediction-e2-20261024.md)の検証を完了。E2は短い予備対照に加えてMNIST4の5,000標本・4 seedまで確認したが、改善が小さく専用枝単独は低下したため、現行形は未採用・本規模化見送り。製品は基準版12のまま。検証・レビュー・数値と採否の正本は各研究記録。E1aの原因の時系列診断、E1b/E3の試作は未実施。研究ツール段階ではspec不要、本実装を決めたらcc-sddというユーザー指示を維持する。1時間以上かかる見込みの実験は55分おきに確認し、早く完了通知が来たらその時点で分析する。

- 作業場所は`.worktrees/refactoring`、ブランチは`refactor/architecture`。mainへはまだマージしない。同じworktreeの主担当は一人にする。
- **研究の現在地と次の一手は[最新の研究引継ぎ（2026-10-26）](../../docs/research/research-handoff-20261026.md)を最初に読む**。10月23日の引継ぎは交代時点の背景資料。
- 直近に完了したspecは[fedsda-loss-monitor-rebaseline-and-all-evaluating-clients](../specs/fedsda-loss-monitor-rebaseline-and-all-evaluating-clients/tasks.md)（基準版13。損失の低下で監視の基準を下げる扱いと、評価clientの上限の撤廃を既定にした）。その前は[fedsda-cross-evaluation-clients-with-target-samples](../specs/fedsda-cross-evaluation-clients-with-target-samples/tasks.md)（基準版12）。
- FedSDAと比較構成、FedDrift、単一モデルのFedAvg（旧Oblivious）の移植・全体run・成果物・掃引入口は接続済み。旧実装は参照として残っている。古い移植順を次の作業と誤認しない。
- FedSDAの既定は配布時モデルとローカル更新モデルの混合予測、帰属保留FIFO30、共通時間尺度T120。Tはモデル別重みと二者混合で共通に使い、両者を別パラメータにしない。
- 2026-10-20から、既定は候補を検証せずにその場で採用し、候補の学習では共有部を固定する（基準版7。[spec](../specs/fedsda-default-adoption-with-fixed-candidate-shared-features/tasks.md)）。それまでの候補検証ありは構成`fedsda-with-candidate-validation`、検証なしで共有部も更新する構成は`fedsda-without-candidate-validation`。候補検証の処理は残している（消すかは、ユーザーが別に決める）。
- 2026-10-20から、クロス評価で配布するモデルを使う（基準版8。[spec](../specs/fedsda-cross-evaluation-with-distributed-models/tasks.md)。精度は同じで、FedSDAの下りの通信量だけが減る。それまでの結果の通信量は旧の扱い）。
- 2026-10-20から、統合で同じ共有部を平均せずに保つ（基準版9。[spec](../specs/fedsda-consolidation-keeps-shared-features/tasks.md)。統合が起きたrunでは値がわずかに変わる。版8のSEAの通信量の増加は、この誤差による共有部の送り直しだった）。
- 2026-10-21から、警報の後の変化点と帰属先を最小記述長の原理で選ぶ（基準版10。[spec](../specs/fedsda-minimum-description-length-alarm-interval-assignment/tasks.md)。設定`alarm_interval_assignment_policy=minimum_description_length`）。保有モデルの再利用の判定に固定の許容量γを使わない。構成`fedsda-with-candidate-validation`は今の扱い（`detector_change_span_and_mean_loss_increase_tolerance`）のまま。サーバのクラスタリングの閾値γ=0.1は残る（[IMPROVE-026](../../docs/research/improvement-candidates/improve-026-server-clustering-by-description-length.md)）。SINE2・CIRCLE2・間隔100の通信の増加（見逃した再利用）は、このspecでは扱っていない。
- 2026-10-22から、「新しいモデル」の仮説を、初期化し直した学習者とwarm startの学習者の短い方＋ln 2で測り（`new_model_hypothesis_learners=reinitialized_and_warm_started_learners`、NEW-006）、初期モデルの損失の統計を事前学習の標本の5分割交差検証で作る（`initial_model_loss_statistics_estimation=five_fold_cross_validation`、LEGACY-020）（基準版11。[spec](../specs/fedsda-warm-started-learner-and-cross-validated-initial-statistics/tasks.md)）。構成`fedsda-with-candidate-validation`は基準版10までの扱い。研究ツールの変種は、`product-default`以外、基準版10の既定を基点にする。
- 2026-10-23から、クロス評価で、対象のモデルを保有し、その標本を返せるclientだけを評価させる（基準版12。`cross_evaluation_client_selection_policy=clients_with_target_model_samples`、LEGACY-022の修正）。構成`fedsda-with-candidate-validation`は今の扱いを明示する。研究ツールの変種は、`product-default`系以外、基準版10の既定を基点にする。
- 動作基準の現在の版は定数`BEHAVIOR_BASELINE_VERSION`と[refactoring-baseline.md](../../docs/experiments/refactoring-baseline.md)の基準版の節が正本（環境別golden）。版13は[fedsda-loss-monitor-rebaseline-and-all-evaluating-clients](../specs/fedsda-loss-monitor-rebaseline-and-all-evaluating-clients/tasks.md)、版12は[fedsda-cross-evaluation-clients-with-target-samples](../specs/fedsda-cross-evaluation-clients-with-target-samples/tasks.md)。版6は[fedsda-default-synchronized-local-mixture-prediction](../specs/fedsda-default-synchronized-local-mixture-prediction/tasks.md)、版7は[fedsda-default-adoption-with-fixed-candidate-shared-features](../specs/fedsda-default-adoption-with-fixed-candidate-shared-features/tasks.md)、版8は[fedsda-cross-evaluation-with-distributed-models](../specs/fedsda-cross-evaluation-with-distributed-models/tasks.md)、版9は[fedsda-consolidation-keeps-shared-features](../specs/fedsda-consolidation-keeps-shared-features/tasks.md)、版10は[fedsda-minimum-description-length-alarm-interval-assignment](../specs/fedsda-minimum-description-length-alarm-interval-assignment/tasks.md)、版11は[fedsda-warm-started-learner-and-cross-validated-initial-statistics](../specs/fedsda-warm-started-learner-and-cross-validated-initial-statistics/tasks.md)、環境と手順は[refactoring-baseline.md](../../docs/experiments/refactoring-baseline.md)。環境差だけを理由にgoldenを更新しない。

## 次の一手

1. 共通引継ぎ手順に従いブランチ・HEAD・差分を照合し、[最新総括](../../docs/research/research-handoff-20261026.md)を読む。必要なresultsはGit管理外なので存在を確認する。
2. 最新のユーザー依頼に従う。E3は提案のみで未選択・未着手。B系列を追加実験する、過去のE1a/S1/S3対照を再実装すると自動判断しない（基準版13は既定。その上で何を試すかはユーザーが決める）。
3. E3が選ばれた場合は、既存IMPROVE-029を出発点に研究ツールの規則・対照を具体化する。手元2500標本以上・4seed、既定との精度/資源比較と診断を併記する方針。本実装すると決まった段階でcc-sddへ進む。
4. 基準版13は2026-10-27に既定にした（上の「現在地」）。候補検証コード削除・main統合・コミット保留資料は未決定。研究ツールと現在の既定を分ける。

## 決定事項

- 研究目標は候補検証の削除自体ではなく、「検証なしの改善を残し、検証ありが防ぐ低下・通信増加をより単純に抑える」こと。検証あり＋新方式も候補。許容精度低下幅は未設定で、自動採否はしない。
- D（待機後強制採用）、R（既存重み比率保持）、P（新IDの初期質量抑制）、共有部25%反映は、一般的な置換方式として採用していない。候補学習中の共有部の固定は、候補検証ありとの組合せ（codexの小規模・研究ツールの本規模とも不安定）では採らず、検証なしとの組合せを2026-10-20に既定にした（本規模・seed 0〜9）。条件限定の改善と一般的な優位を区別する。
- 探索用の対照は`tools/research_*`等へ隔離し、共通アルゴリズム・設定選択肢・通常pytest/CI/freshへ枝を増やさない。研究専用checksは明示実行する。通常入口へ採用済みの再開補助・メモリ回収・seed構文は、この研究分岐の扱いとは区別する。
- 標準JSON/NPZ/READMEの形式は再開補助・省メモリ化・seed構文のためには変更していない。研究診断用の別形式JSONは標準run形式と混同しない。
- 旧実装との一致を守る参照設定と、改善した新既定を区別する。旧結果の一括変換ではなく、新実装でablationを実行する方針を維持する。
- レビュー手順は共通引継ぎ手順を参照する。全pytest、および権限で実行不能なfresh smokeは、独立レビュワーが主担当の実測証拠を照合し、独立未実行を明記して判定できる。過去の子エージェントIDは来歴であり、Claudeから再利用する接続先ではない。

## 研究記録の入口

- [要素ablation](../../docs/research/fedsda-component-ablation-20261017.md): 720runの条件別比較。検証の有無はデータ・間隔で符号が変わる。「ローカル更新モデルのみ」と「現在の学習帰属モデルのみ」は別方式。
- [候補検証の原因診断](../../docs/research/fedsda-candidate-validation-diagnosis-20261017.md): MNIST2/4、共有部置換、予測重みなどの初期切分け。
- [非MNIST共有部反映](../../docs/research/fedsda-candidate-backbone-blend-nonmnist-20261018.md)、[候補共有部固定144run](../../docs/research/fedsda-frozen-candidate-backbone-20261018.md): 単純な共有部保護だけでは全条件を改善できない。後者の間隔200は現行処理どおり2400標本評価（生成2500）。
- [単純化の整理](../../docs/research/fedsda-candidate-validation-simplification-20261018.md): 問題の分解と次の仮説。5seed時点の表なので最新のV/O判断は次の10seed報告も参照する。
- [サーバ3実験・10seed比較](../../docs/research/fedsda-server-results-20261018.md): 新規660runと既存対照の受領・照合。検証なしはMNIST2の全間隔で平均低下し、MNIST4・500では改善。通信は24条件平均すべて増加。混合予測とFedDrift/FedAvg比較も含む。
- [D本規模54run](../../docs/research/fedsda-candidate-waiting-diagnostic-20261018.md): 待機後強制採用はV/O両方を上回る代替にならず、共通経路から分離済み。
- [R/P本規模72run](../../docs/research/fedsda-prediction-initial-mass-n5000-20261019.md): 予測初期質量だけの制御では保護効果を代替できない。学習・適応・通信をOと揃えた対照。[実行手順](../../docs/experiments/prediction-initial-mass-diagnostic.md)。
- [uranium回収対照](../../docs/research/fedsda-live-reclamation-uranium-20261019.md): 学習結果を変えずにメモリを減らす実測と適用範囲。現在は通常FedSDAへ接続済み。

## 採用済みの実行補助と環境の注意

- 掃引の再開: 同条件の過去出力と進捗から再開先を選択できる。完全な既存runはスキップし、完了済みを選ぶと実行を終える。中断runは補助証明と条件・コード・環境が一致する場合だけ退避して再実行する。証明のない旧不完全ファイルや破損・不一致を盲目的に上書きしない。[手順](../../docs/experiments/experiment-sweep.md)。
- seed: `--seeds 0-9`は両端込み。`--seeds 0-2 5 8-9`の列挙混在も可。負数・逆順・重複・範囲重複は拒否する。間隔/T/FIFOのステップ付き範囲やdatasetの`all`指定は未実装。
- メモリ: Linux/glibcの通常FedSDA全体run・掃引は、正常な同期末に`malloc_trim(0)`を呼ぶ。Windows等の非対応環境はno-op。FedDrift/FedAvgや低レベルstream単独には接続していない。[運用と制限](../../docs/experiments/fedsda-runtime-memory-reclamation.md)。
- uraniumでのMNIST2・seed0・1workerの1対では最大RSS81.54%減、時間1.79%増、全指標・133配列等一致。32/64workersでの導入後の同条件対照は未実施。この削減率から並列数をそのまま算出しない。アルゴリズム・標本表現・allocator設定の一律変更はしていない。
- 過去の独立回収prototypeは旧source hash固定の研究再現用。通常実行は最新ブランチを取得する。uraniumが一時detached HEADだったためブランチ復帰手順を案内済みだが、サーバ側更新の完了は未確認。
- サーバ実行と転送はユーザー担当（踏み台経由）。長時間実験はLinux `.venv`・tmux、各数値ライブラリ1thread。uraniumのコア数だけでworkerを増やさず、メモリ・負荷を確認する。
- Windows検証Pythonは元checkoutの`venv/Scripts/python.exe`。Linux検証WSLは`Ubuntu`を明示する（既定がdocker-desktopになっていた）。MNIST入力・固定環境は基準文書に従う。
- 研究入力・生の結果・一時スクリプトの多くはGit管理外。`results/`が存在するこのworktreeと、Gitだけをcloneした環境は同じではない。各報告の入力パスと完全性を確認し、一括stageしない。

## 未解消の事項・次の候補

- 候補検証を単純化しV/O両方に優位となる方式として、検証なし＋候補の共有部の固定を採用した（[研究記録](../../docs/research/fedsda-provisional-candidate-expert-20261019.md)）。検証なしのMNIST2の低下が共有部の上書きから来るという説明は、固定で低下が消えたことと合うが、直接の切り分けではない。
- 純粋予測方式seed5〜9の240run、手法比較（間隔100/500）seed5〜9の180runは補完コマンドを提示済みだが、実行・結果受領は未確認。確認なしに完了扱い・再発注しない。
- LEGACY/IMPROVE/NEW/UNPORTEDの現在の採否は[実装findings](../../docs/research/implementation-findings/README.md)と[改善候補](../../docs/research/improvement-candidates/README.md)が正本。10/13以降の修正・採否を、古い「残るLEGACYをまとめて判断」へ戻さない。
- LEGACY-001とLEGACY-014の概念計数は旧実装を外した後に直す判断、NEW-004は旧依存の計算量照合を外す準備の記録。LEGACY-019の改名参照除外の頻度・影響は未確認。詳細と変更時期は各記録を参照する。
- 未移植の細目はUNPORTED-001（検出episode）、002（参照も学習する検証）、003（回復期限定の混合）。現在の不具合や移植漏れと自動判定しない。その他の診断追加や旧実装削除は、目的・依存を確認してから仕様化する。
- 一部の古いspecはtask完了でもphaseが`implementation-complete`等のまま。履歴上の状態表現と直近specの未完了を混同せず、spec.jsonとtasksの証拠を読む。
- 動作図HTMLは別ブランチ`docs/fedsda-processing-flow`で整理・push済み、専用worktreeは削除済み。mainへのマージは保留。refactoringへ自動コピー・混在させない。

## 直近・主要な完了spec

- [fedsda-consolidation-keeps-shared-features](../specs/fedsda-consolidation-keeps-shared-features/tasks.md): 統合で同じ共有部を平均せずに保つ扱いの既定と基準版9。
- [fedsda-cross-evaluation-with-distributed-models](../specs/fedsda-cross-evaluation-with-distributed-models/tasks.md): クロス評価で配布するモデルを使う扱いの既定と基準版8。
- [fedsda-default-adoption-with-fixed-candidate-shared-features](../specs/fedsda-default-adoption-with-fixed-candidate-shared-features/tasks.md): 候補検証なし＋候補の共有部の固定の既定と基準版7。
- [seed-range-command-arguments](../specs/seed-range-command-arguments/tasks.md): seedの両端込み範囲と列挙混在。
- [fedsda-runtime-memory-reclamation](../specs/fedsda-runtime-memory-reclamation/tasks.md): 通常FedSDAの同期末メモリ回収。
- [run-memory-reclamation-prototype](../specs/run-memory-reclamation-prototype/tasks.md): 同期末回収の独立対照ツール。
- [prediction-initial-mass-diagnostic](../specs/prediction-initial-mass-diagnostic/tasks.md): R/Pの独立研究診断。
- [candidate-validation-waiting-diagnostic](../specs/candidate-validation-waiting-diagnostic/tasks.md): 待機後強制採用Dの研究診断。
- [experiment-sweep-resumption](../specs/experiment-sweep-resumption/tasks.md): 過去出力の選択・再開・安全復旧。
- [fedsda-component-ablation-analysis](../specs/fedsda-component-ablation-analysis/tasks.md): 要素ablation解析と人による採否判断の補助。
- [fedsda-default-synchronized-local-mixture-prediction](../specs/fedsda-default-synchronized-local-mixture-prediction/tasks.md): 混合予測・FIFO30・共通T120の既定と基準版6。
- [fedavg-experiment-run](../specs/fedavg-experiment-run/tasks.md): 単一モデル比較手法FedAvgの移植。
- [feddrift-experiment-sweep](../specs/feddrift-experiment-sweep/tasks.md): FedDriftの全体run・成果物を掃引入口へ接続。
- [important-behavior-regression-goldens](../specs/important-behavior-regression-goldens/tasks.md): 新実装の重要経路を環境別goldenで固定。

上記は直近と主要な到達点の索引で、全specの一覧ではない。以前の移植は[roadmap](roadmap.md)と各specを参照する。完了specのレビュー経緯・検証件数・commitは各tasks.mdのImplementation Notes等へ集約し、ここへ複製しない。
