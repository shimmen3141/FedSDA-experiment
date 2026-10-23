# S1の無変化・オンライン継続対照（2026-10-24）

## 目的と今回の範囲

[成立性診断](fedsda-shadow-learner-s1-diagnostic-20261024.md)でSEA4概念2の24影中8件が初期差を回収した。ただし他概念でも交差し、実際の採用の有効性は不明。ユーザー指示により次の対照を研究ツールとして準備した。specは不要という指示を維持し、製品src・golden・採用の処理は変更しない。

1. **無変化対照**: SEA4の概念0/2、SINE2の概念0/1に全clientを固定する。事前学習も同じ概念から生成し、評価開始時に人工的なドリフトを作らない。元の概念スケジュールの生成を先に実行してPython乱数の消費順を維持し、列だけ置換する。これらは元ベンチマークの精度比較とは別の診断用入力。
2. **現在のモデルのオンライン継続対照**: 警報の処理後の現在の分類器と概念固有optimizerを一緒にdeepcopyする。以後、影と同じ標本で学習前の損失を測り、毎標本1回・共有部固定で概念固有部だけ更新する。元の現在のモデルは通常どおり動かす。パラメータとoptimizerの対応・状態は引き継ぐ。

記録する値は次のとおり。

- D = 初期差 + Σ（製品の現在のモデルの損失 − 影の損失）。従来の診断と同じ。
- C = 初期差 + Σ（現在の写しをオンライン継続した対照の損失 − 影の損失）。同じ更新手順でも影が有利かを見る。
- A = Σ（製品の現在のモデルの損失 − オンライン継続対照の損失）。D = C + Aを全標本で検算する。

Cは診断用の差で、MDLの新しい採用規則ではない。影と継続対照は、初期状態・過去の標本・optimizerの履歴が違うため、交絡を完全に除去した比較でもない。D/Cが同時に正になった件数も記録する。両方の交差は、開始時の真の概念が一度でも変わった後のものと区別する。真の概念は層別用のみで、モデルの生成・更新・終了には使わない。

各条件で、同じ入力・環境の既定を実行してから診断を実行する。秒数と追加計算以外の全保存値を照合し、追加モデル計算を差し引いた計数も一致を要求する。元の影と継続対照の学習は合計で1観測2回。追加推論も全て計数に含む。

## 準備と手元検証

専用入口は`tools/research_shadow_learner_diagnostic/run_controls.py`、分析は`analyze_controls.py`、Linux用は`run_controls.sh`。従来の診断はオンライン継続対照の引数を省略すると元の経路を使う。先行する結果の実行時ソースは、変更前のsnapshotとhashを保持した。

- 部品・境界の関連検証: **60 passed / 2 deselected、2.62秒**。固定概念の生成、事前学習側にも固定が適用されること、スケジュール乱数の消費、context終了時の復元、optimizerの参照対応と状態継続、製品側の状態・乱数不変を含む。
- テスト初回は、テスト側が必須引数のある`RandomConceptScheduleSettings`を引数なしで生成して4件失敗した。実際の型宣言に従って条件を明示し修正した。製品や診断の不具合ではなかった。
- 全体確認: 6入力・seed1・600標本・間隔100、各入力で既定と診断の計12 run。**6条件成功・失敗0**。全保存値と追加計算を差し引いた計数が一致。分析で全軌跡のD/C/A・交差・2回/観測の学習を検算した。
- Ruff E4/E7/E9/F成功。製品の全pytestやgolden再生成は実施していない。
- 独立CLIレビュー: `gpt-6-luna`、effort medium、**PASS**、session `01a16a2f-f7b3-7652-a46e-5796a3e6775d`。6.1-lunaは先行作業で利用不能。読取りレビューで、試験は独立未実行。ログは`results/_sol_prompts/s1-controls-review.txt`。
- Linux用シェルはGit Bashの`bash -n`成功。Linux上でのwrapper全体実行は未実施。実行手順とアーカイブの独立確認（session `01a16a33-5c37-7110-87d8-d0c78ae334ff`）ではtime出力の親ディレクトリに指摘があった。元のsidecarも出力と同じ親なので不足はなかったが、timeとlogの変数と各親の作成を明示した。再レビューは**PASS**（`gpt-6-luna` medium、session `01a16a34-ac9b-7540-ae62-9c020b2db1b1`）。いずれも読取りのみで独立実行は無し。
- 転送用ZIPの全244ファイルをmanifestのSHA-256と照合し、展開した独立snapshotからSEA4概念2固定・100標本の既定/診断も実行して保存値一致を確認した。ZIPは以後の文書・シェル更新を含めて再生成し、全entryのhashを再照合した。

短い確認の出力は`results/fedsda-shadow-controls/20261024_smoke-n600/`。無変化でもSEA4概念0は21影中2件、概念2は42影中5件でD>0。オンライン継続対照にも同時に勝ったのはそれぞれ2件。SINE2概念1は5影中2件でD>0だがC>0は0件。SINE2概念0では影が0件だった。この1 seed・短い確認だけで採否は決めないが、無変化対照の必要性を裏づける。

## uraniumで実行する本規模の条件

| 軸 | 値 |
| --- | --- |
| 入力 | SEA4の通常ドリフト、概念0固定、概念2固定 / SINE2の通常ドリフト、概念0固定、概念1固定 |
| 集約間隔 | 100、500 |
| seed | 0〜9 |
| 生成標本数 | clientあたり5,000、client数10 |
| 影の観測上限 | 240（2T） |
| 条件数 | 120、各条件の既定と診断で計240 run |
| 並列 | uranium（256コア）の標準64 workers、数値ライブラリ各1thread |

通常ドリフトにも継続対照を足すので、前回の診断と重複する実験ではない。対照の既定も同じLinux環境で取り直す。実行前のprint-planで120条件・240 run・未使用の出力先を確認し、確認だけではディレクトリが作られないことを確認した。本規模はまだ実行していない。サーバの実行速度・メモリで所要時間が変わるため、完了時刻は未見積り。1時間を超える場合の確認は55分間隔というユーザー指定を維持する。

サーバでの実行はユーザー担当（踏み台経由という既存の分担）。2026-10-24の追加指示で研究ツールのpushが許可されたため、uraniumの既存checkoutでブランチを更新して使う。ZIPによる独立snapshotも残すが、今回の通常手順には不要。結果・raw・レビューの生ログはGitに含めない。

uraniumのリポジトリルートで実行する。未コミット変更があってswitchやpullが失敗した場合は、変更を破棄せず状態を確認する。

```bash
git fetch origin
git switch refactor/architecture
git pull --ff-only origin refactor/architecture
source .venv/bin/activate
tmux new-session -s fedsda-s1-controls
```

開いたtmux内で、既存の研究CLIと同じく条件とworkersを明示して実行する。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m tools.research_shadow_learner_diagnostic.run_controls \
  --output results/fedsda-shadow-controls/20261024_uranium-s1-controls-seeds0-9-n5000 \
  --intervals 100 500 --seeds 0 1 2 3 4 5 6 7 8 9 \
  --sample-count 5000 --horizon-samples 240 --workers 64
```

成功後に同じ出力先を分析する。

```bash
python -m tools.research_shadow_learner_diagnostic.analyze_controls \
  results/fedsda-shadow-controls/20261024_uranium-s1-controls-seeds0-9-n5000
```

6種類の入力（通常ドリフトと4種類の固定概念）はこのCLIに固定されている。直接CLIも`--workers`で任意の正の並列数を指定でき、14の上限は無い。条件・ソースhash・対照/診断の結果・JSONLを保存し、失敗はfailures.jsonへ残す。既存の出力先には上書きしない。

日時つき出力先・ログ・GNU time・成功後の分析までまとめて保存したい場合は、補助shellも使える。

```bash
FDE_WORKERS=64 bash tools/research_shadow_learner_diagnostic/run_controls.sh
```

補助shellのuranium向け既定も64へ訂正した。古い14という案内は手元の目安との混同だった。環境別の現行条件と古い記述の所在は[steeringのtech.md](../../.kiro/steering/tech.md)を参照する。旧Pareto CLI専用の`tools/run_server_sweep.sh`を全ての研究CLIへ要求しない。shellの出力は`results/fedsda-shadow-controls/<日時>_uranium-s1-controls-seeds0-9-n5000/`で、結果と隣の同名.execution.log・.time.txtを一緒に回収する。直接CLIの場合は指定した出力ディレクトリを回収する。

## 結果が返った後の判断

無変化と通常ドリフトをseed単位で比較し、警報数・影の数だけでなく、Dのみの交差とCにも勝つ交差、打切り理由、初期差・軌跡、追加計算を見る。影0件のseedも分母から落とさない。影の反復を独立試行として有意差を作らない。D>0をそのまま採用してよい根拠にはしない。今回も採用後の精度・過剰分離・サーバ統合後の保持は測っていない。本実装へ進む判断は未了であり、その場合はcc-sddを使う。

結果待ちの手元作業では、採用機能の実装に進まず、seedごとの対照集計と判断基準の整理を優先する。無変化でDのみ交差するなら更新手順の差、Cにも勝つなら初期化・履歴・ラベル雑音等の影響を調べる必要があり、どちらも即座に誤採用と断定しない。通常ドリフトの概念2で複数seedに共通する優位が見えるかを合わせて見る。採用を試す段階では初めて精度・通信・統合後の保持を測り、それまでは本実装の仕様を確定しない。
