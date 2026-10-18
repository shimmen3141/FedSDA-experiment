# FedSDAの省メモリ調査（2026-10-19）

## 目的と判断の境界

uraniumの多コアを利用するため、計算結果・乱数消費・学習/検証/統合の判断・記録形式を維持してメモリだけを減らせるか調べる。ユーザーから受領した調査メモの6.76 GiB/runは、こちらで再計測した値ではない。

ユーザーからの追加報告により、6.76 GiBの出所は確定した。uraniumで元の`run_experiment_sweep.py`をMNIST2・seed0・10client・5000標本/client・間隔50・workers1で実行したGNU timeの`Maximum resident set size (kbytes): 7088196`である。コマンド全体の最大RSSで、ホスト使用量をworker数で割った推定値ではない。全並行runがMNIST2であることは並列時の総使用量の説明にはなるが、この単独runの値の説明にはならない。

まず実装を変更せず、保持オブジェクトと解放済み領域の滞留を分けて測定した。新しい製品オプションや共通フローの分岐は追加しない。この文書は実装specではなく、実装の採否に先立つ調査記録。

**現時点では製品コードの改修を見送る。** 省メモリの余地自体はあるが、uraniumの6.76 GiBを手元で再現せず、ピークに効く主要な保持対象は未特定。局所的な回収対策もピーク削減が小さいか、所要時間増加を伴った。次はuraniumで同じ計測を行う価値が高く、現在の証拠だけで保持契約を横断して作り替える価値は低い。

## 再現条件と証拠

- ソース: `c6a1720a8ba925efb628c39b0e15ae7af1efc7d9`。`src`の内容hashは`3861f16d77fad2449baf39e37c433bb5343b56f4291efa315ce9c3aea8cf71e2`。
- 受領メモのサーバcommit `8ace43cefd5cd0acf77d1babccf3df9903ab9c37`から現在まで、`git diff ... -- src`は空。
- 手元Ubuntu/WSL2、Python 3.14.4、NumPy 2.4.6、torch 2.12.1+cpu、glibc 2.43。uraniumのPython 3.10.16/NumPy 2.2.6とは異なる。
- 実行入口も異なり、uraniumは元の掃引CLI、手元は診断scriptから製品関数を直接呼んだ。1.69 GiBは最初のrun返却まで、6.76 GiBはコマンド全体。手元も保存後まで約1.77 GiBなので、期間の差だけで説明できたとは扱わない。環境・入口・実際の学習経路を揃えた比較が必要。
- `fedsda`、MNIST2、seed0、10client、間隔50、1000または5000標本/client。設定は通常の`build_experiment_run_settings`から生成し、既定値を変更しない。
- OpenMP/MKL/OpenBLASは各1、`torch.set_num_threads(1)`。複数の計測プロセスを並行した。所要時間は同じ負荷条件での性能ベンチマークではない。
- 一時script、元NPZ/JSON、段階別RSS/PSS、所有量の一覧、照合結果は`results/fedsda-memory-diagnosis/20261019/`。git管理外の研究成果物。

`profile_run.py`は製品関数をそのまま呼び、観測列生成と10ラウンドごとの同期前後を記録する。別スレッドで0.2秒間隔のRSSも読む。最大RSSにはLinuxのhigh-water値を使い、保存後だけに発生した増加とrun中の値を区別する。計測自体にも小さな割当があるため、絶対値は計測付きrunの値。

`inventory`はrun結果から辿れるコンテナ、プロジェクトobject、nn/optimizer object、Tensor storageをidentityで重複排除する。Python浅いサイズの合計とstorageの実データ量を別に数える。Tensorのgrad/autograd内部、ランタイム全体、モジュールのglobal cacheを完全に数えるものではない。これらの合計をRSS全体の内訳とみなさない。

## ソースで確認した所有関係

| 対象 | 確認できたこと | この調査での意味 |
| --- | --- | --- |
| MNIST元画像 | readonly uint8、ディレクトリ別のプロセス内cache | 60000画像の画素部分は約45 MiB。共有化だけで数GiBは消えない |
| 観測特徴 | 正規化済み256個のfloatを再利用する784要素tuple | 50000標本のtuple参照部分だけで約299 MiB。画素ごとのfloat新規生成ではない |
| 学習標本store | recordとTensorを借用し、snapshotもpayloadを共有 | snapshotのたびに画像Tensor全部がコピーされる構造ではない |
| 同期履歴 | 登録/集約/配布/再較正の結果、クロス評価のスカラー・個数記録 | 全ラウンドのモデル重みを保持するrecordではない。二次的な記録量は実測する |
| 予測用の配布時モデル | 現在の保有モデルを一括deepcopyし、前回の写しを置換 | 必要な2状態の保持。過去ラウンド全部の保存ではない |
| 計算量meter | パラメータのweakrefと計数 | 入出力Tensorやモデルを強参照で蓄積する実装ではない |
| sweep worker | spawn、worker内で保存、小さい結果だけ親へ返却 | 親が全runの参加者/モデルを集める構造ではない |

主な照合先は`data/mnist/`、`learning/training/model_training_sample_store.py`、`runtime/server_round_synchronization.py`、`runtime/model_cross_evaluation.py`、`methods/fedsda/prediction_combination/synchronized_prediction_model_snapshot.py`、`learning/models/model_computation_measurement.py`。

## 完了済みの1000標本対照

| 対照 | 1run目のピークRSS MiB | run返却時RSS MiB | run実行秒（参考） |
| --- | ---: | ---: | ---: |
| 変更なし | 1145.94 | 1043.40 | 158.05 |
| 同期終了ごとに`malloc_trim(0)` | 1127.76 | 638.72 | 196.56 |
| 起動時`MALLOC_TRIM_THRESHOLD_=131072` | 980.02 | 683.95 | 366.96 |

後二者は独立した対照で、設定値と明示trimを併用していない。明示trimは一時script内のみで、標本やモデルへの参照を捨てない。

allocator設定による時間増加は採用保留の理由の一つだが、負荷を揃えた対照ではないため、設定による性能悪化率は確定していない。

表のピークは、各プロセスの開始から**最初の`run_return`まで**に記録した`VmHWM`の最大値で、import・初期準備を含む。保存、所有量調査、2run目を含めない。`analyze.py`の`first_run_process_high_water_mib`が対応する。`high_water_mib`と最終`sampled_peak_rss`は2run目や保存も含むプロセス全体の値で、表との比較には使わない。baseline1000の2run全体のhigh-waterは1174.52 MiBだが、1run目返却時までの最大は1145.94 MiBである。

比較は全133配列の列名・shape・dtype・bytes（NaNと符号付き0を含む）、全指標・設定、最終サーバモデルの全パラメータhashが一致した。外側の乱数や内部の乱数源の全状態を別途記録した検証ではなく、この一致を将来の全条件の同一性保証に一般化しない。

変更なしrunで、返却された結果から辿れるPython objectは約72.14 MiB、Tensor storageは約148.12 MiB/20424 storage。tupleは約60.66 MiBで最大のPython種別。Tensor grad等を除く計数である。保存用133配列の非圧縮データは約1.03 MiB。少なくともこの条件では、保存用配列の生成が数GiBの原因という説明には合わない。

参照解放後はRSS980.0 MiB、GC後も同値、`malloc_trim`後399.37 MiBだった。この時点では約580.6 MiBが解放済みでもOSへ返っていなかったことを示す。これはrun中ずっと同じ量が不要だったという証拠ではない。同期末trimの対照では定常時の滞留を減らせても、ピーク削減は約1.6%にとどまった。

`baseline1000`は2runの間にGC/trimを入れた診断でもあるため、通常のworker再利用によるメモリ推移の証拠には使わない。通常の再利用は`reuse1000`として別に計測する。

## 5000標本とworker再利用

通常のworker再利用（強制GC/trimなし）の1000標本2runでは、run返却時RSSが1056.64→1058.73 MiB、参照解放後が974.37→982.41 MiBだった。2run目終了までのhigh-waterは1226.44 MiB（1run目1160.44 MiB）。所有object数173370、Tensor storage数20424・155314168 bytesは両runで同じで、Python浅いサイズ合計は75639415→75609767 bytesだった。全133配列・指標・設定・最終パラメータは変更なしの対照と一致した。大きな逐次蓄積は観測しなかったが、2runで長期のリークを否定はできない。

`MALLOC_TRIM_THRESHOLD_=131072`の5000標本対照は、小規模の完了済み対照で時間増加を確認したため途中で打ち切った。`threshold5000/status.json`に記録し、未完了の値を5000標本の削減率や同一性の証拠に使わない。

5000標本の変更なしrunは完了した。run実行1003.54秒、最初のrun返却までのhigh-waterは1730.95 MiB（約1.69 GiB）、返却時RSSは1718.87 MiB。所有量の走査自体で追加割当があり、保存後までを含むプロセス全体のhigh-waterは1809.57 MiB（約1.77 GiB）になった。後者を学習中のピークと誤記しない。

返却結果から辿れるPython objectは351.40 MiB、Tensor storageは275.99 MiB/100552 storage。そのうちtupleは302.95 MiB。Tensor objectの数は100032で、スカラーや画像の小さなTensorの管理コストも別にある。前述の限界があるので、残りのRSS全部をallocator保持と断定しない。

保存用全133配列は約4.38 MiB。所有量調査後のRSS1778.63 MiBに対して、配列生成・保存後1778.64 MiBで、今回の記録形式の保存が大きな追加ピークの原因とは認められなかった。参照解放・GC後1664.77 MiB、trim後440.21 MiBまで減ったが、これもrun終了後の回収であり、1run中のメモリが1.2 GiB減るという結果ではない。

合計7runを完了（1000標本6run、5000標本1run）、未採用設定の5000標本1runを打切り。1000標本6runは5組の照合で全保存値が同一。5000標本の変更前後の数値対照を完了したとは扱わない。uraniumの32/64workerによるPSS合計・run/時は手元では未測定。

## 次にuraniumで測るもの

uranium上で、次の順に環境差と実行入口の差を切り分ける。各測定は並行させず、同じsource・ライブラリ・スレッド設定を使い、実行時の条件と出力を記録する。

1. **元のCLIを再実行する。** `run_experiment_sweep.py`の元コマンドで、`fedsda`、`--datasets mnist2 --seeds 0 --client-count 10 --per-client-sample-count 5000 --aggregation-intervals 50 --workers 1`を揃え、GNU timeでコマンド全体を測る。既存結果の再開・実行スキップでは再測定にならないため、新規出力を明示し、約6.76 GiBを再現するか確認する。
2. **同じ環境・条件で診断scriptを実行する。** 下の例で、段階別のRSS/PSSとコマンド全体の最大RSSを取得する。元CLIとの完全設定・全指標・記録配列の列/shape/dtype/値を照合し、同じ学習経路を比較しているか確認する。環境の異なる手元結果をこの一致の基準にはしない。
3. **差の位置に応じて調査する。** 両入口とも高ければ増加段階と所有状態を調べる。元CLIだけ高ければ、入口の準備・保存・オブジェクト寿命・親/子processの寄与を優先する。元CLI自体が再現しない場合は、実行条件・環境・負荷の差を先に確認する。

今回の一時scriptと`identity.json`を含む研究ディレクトリをサーバへ移す。scriptはgit管理外なので、git pullだけでは取得できない。以下は手順2の実行例で、手順1の元CLI再現を代替しない。

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /usr/bin/time -v python results/fedsda-memory-diagnosis/20261019/profile_run.py \
  --samples 5000 --name uranium-baseline5000
```

一意な出力名で実行する。製品の既存結果は読替え・上書きしない。MNISTが既定の配置にない場合は既存の`FDE_MNIST_DATA_DIR`を指定する。この一時scriptは独立した1run診断であり、掃引や採否を実行しない。

最優先は6.76 GiBになる段階（準備、観測列、同期中、返却、保存）と、所有するPython object/Tensor storageの量を特定すること。その後にだけ最大の対象の寿命・格納方法を試作し、同じuranium環境で全133配列・指標・最終パラメータを照合する。起動時allocator設定の無条件推奨や、32/64workerのいきなりの増員はしない。

## 実装候補の選別

- **学習データの間引き、FIFO/評価/検証数の縮小、float16、batch変更、診断の削除は対象外。** 結果を変え得る。
- **無条件のモデルコピー削除は採らない。** 配布時と更新用の2状態は混合予測に必要。現在の履歴構造にも全モデルコピーの蓄積は見つかっていない。
- **標本ごとのGC、共通フローへの実験専用分岐は採らない。** 今回のGCだけではRSSが戻らず、広範囲に手を入れる根拠がない。
- **標本表現の圧縮は優先しない。** tupleを配列へ変えるにはexact-typeの公開契約、結果保持、snapshot、入力変換を横断する。省メモリ余地はあるが、報告された6.76 GiBの主因と確認できていない。
- **allocatorの回収は有力な診断対象だが、そのまま既定にしない。** 終了時RSSの削減とピーク削減は異なり、小規模では起動時設定の所要時間増加も観測した。uraniumでの同一条件対照が必要。
- **workerのrun単位終了はrun間だけの対策。** 単一runのピークを減らさず、import・準備のコストもある。必要性を再利用測定から判断する。

## 候補検証の研究に戻る場合の次案

[Pの72run](fedsda-prediction-initial-mass-n5000-20261019.md)では新IDの予測初期質量だけでは低下を解消しなかった。Dによる一律待機、共有部25%反映、候補共有部固定も全条件を支配していない。次の一律係数の掃引より、同じ将来標本で次を分けて観測するのを優先する。

1. **候補headの利得と共有部コピーの害を別に測る。** 共有部変更前後の既存headの出力と、候補headの出力を対照化し、既存モデルへの干渉と候補自体の良否を区別する。共有部を固定すればよいと先に決めない。
2. **接続後の実際の混合をshadowで比較する。** 既存の混合と候補追加後の混合を、判断にまだ使っていない将来標本で対照化する。候補単体対固定参照の勝敗と、最終出力への寄与が一致するかを調べる。
3. **ローカル採用とサーバ共有を分ける案を通信対策として検討する。** 即時適応の利得を残し、役に立たないモデルの共有・再配布を抑える。ただし、共有前のローカル精度低下まで治すものとは期待しない。

いずれも未実装の仮説。まず1/2を独立した観測scriptで診断し、利得/害の場所が分かってから制御ロジックを決める。候補検証の既定や採否基準は変更しない。

## 検証と独立レビュー

- 主担当: 完了した5プロセスはexit0（1000標本6runと5000標本1run）。起動時設定の5000標本対照は主担当のSIGINTによる打切り、exit1、結果なし。未完了を成功に数えない。
- `analyze.py`の5比較はすべて133配列bytes・設定/指標/最終パラメータhashが一致。学習の独立再実行や全条件での同一性保証ではない。
- 初回の独立CLI GPT-6 Luna high: session `01a14fad-3b67-7e93-84c0-c4198de92002`、FAIL。ピークの期間定義を明示するMajorと、再利用結果/打切り理由の記載に関するMinorを採用。元の測定値は変更せず、最初のrun返却までのhigh-waterを集計する項目と説明を追加した。
- 修正再確認の独立CLI GPT-6 Luna high: session `01a14fb2-042a-7ab3-bb91-3bebb9439074`、PASS、残る指摘なし。初回がephemeralだったため前回指摘を引き継いだ新規CLI sessionで限定再確認した。両方ともログの実モデルとeffortを確認。
- 初回は保存値の5比較、代表的な所有関係、source差分が空であることを独立照合。再確認はピークの定義と5000標本の計数・換算を独立照合。学習・pytest・uraniumの計測は独立未実行。製品src・既定・golden・保存形式の変更はなく、全pytestは再実行していない。
- レビューログは作業rootの`venv/refactoring-tests/memory-diagnosis-{method,final}-review.log`。一時script・元記録の証拠は前述のresults内に保持する。
- 追加報告の反映: 6.76 GiBのGNU time出力と元条件をユーザーの報告として追記し、uraniumの手順を元CLI再現→同環境の診断script比較へ明確化した。主担当によるサーバ再実測や追加の独立レビューを行った記録ではない。既存の測定値・過去の承認記録は変更していない。
