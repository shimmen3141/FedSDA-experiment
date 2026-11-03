# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedSDAの手法を論文（FedHASP、PAKDD 2027）にまとめる研究者（ユーザー）。
- 今の状況: CFL+Window（`feddrift-cfl-window`、`.kiro/specs/feddrift-cfl-window-configuration/`）を1時刻1roundで予備の実行（24 run）をしたところ、分割が一度も起きず、FedAvg+Windowと同じになった。CFLの分割の条件は「平均の更新が観測した最大の1/10未満」で、同じ時刻のデータを何度も学習して収束することを前提にしている。FedDriftの論文は全手法を1時刻100 round（合成・MNIST）、公式の実行例のCFLは1時刻200 roundで動かしている。
- 2026-11-03のユーザー判断: 手法は機能する形で比べる。round数を増やすと通信・計算が増えるが、精度と通信量のトレードオフ（論文の図3）で測れるので、CFL+Windowのround数を増やした版を比較に入れてよい。公式の実装にできるだけ合わせ、設定の違いから来る差（全手法に共通の条件）は許容する。精度の差がクラスタリングによるのか学習の回数によるのかを分けるため、同じround数のFedAvg+Windowを対照に置く。
- 何を変えるか: 1時刻あたりの学習・集約のround数 $R$（既存の`FeddriftServerSettings.local_training_round_count`、既定1）を、FedAvg+Window（`feddrift-window`）とCFL+Window（`feddrift-cfl-window`）の2構成で1より大きくできるようにし、掃引の値として指定できるようにする。CFL+Windowは、時刻の途中の分割の後の扱いを公式に合わせる。

## Boundary Context
- 対象: 上の2構成の $R>1$、掃引の値、CFLの時刻の途中の分割の扱い、記録の列、test、文書、予備の実行のコマンド。
- 対象外: ほかの構成の $R$（IFCA+Windowは1時刻1roundのまま。FedDrift本体などの既存の挙動も変えない）、既存の構成の設定・値・runの識別子・golden、局所の学習の予算・学習率・事前学習の初期モデル。

## Requirements
### Requirement 1: 1時刻のround数の設定
#### Acceptance Criteria
1. The system shall 掃引の値`--feddrift-local-training-round-counts`（正の整数の列）で、`feddrift-window`と`feddrift-cfl-window`の1時刻のround数 $R$ を指定できる。既定は1で、既定のrunの設定・識別子・goldenは変えない（既定値は保存設定から省くか、既存の識別子が変わらない形にする）。
2. The system shall ほかの構成で既定以外の $R$ を指定したとき、組み立ての前に拒否する（掃引の計画と、runの組み立ての検査の両方）。組み立ての検査のエラーの文言は、拒否する構成を正しく言う（今の文言は「IFCA+Windowの」のままである）。
3. The system shall $R$ 回のroundのそれぞれで、今と同じ局所の学習の予算・学習率・件数の条件・集約の重み（その時刻の窓の件数）を使い、学習・上りと下りの通信・計算の計数を毎round行う。$R$ を増やすと学習と通信はround数に比例して増える。
4. The system shall 窓の標本を時刻の初めにだけ空にし、その時刻の $R$ 回のroundは同じ窓の標本で学習する（公式の`win-1`と同じ）。

### Requirement 2: CFL+Windowの時刻の途中の分割
#### Acceptance Criteria
1. When $R>1$, the system shall 公式どおり、毎roundで、clientの学習の後・集約の前に、そのroundの学習の後の更新で分割を判定する（判定の規則・状態・2分割・上限は既存の`CflClusterSplittingState`のまま）。
2. When 時刻の途中のroundで分割が起きる, the system shall そのroundの集約を省き、分割したモデルと新しいモデルを $\theta_0$ にし（既存どおり）、片方のclient群を新しいIDへ割り当て直す。さらに、その時刻の残りのroundでは、割り当て直したclientのその時刻の窓の標本を新しいモデルの学習の標本として使い、元のモデルの学習の標本から除く（公式の`train_data_weights[curr_iter]`の書き換えと同じ）。
3. The system shall 学習の対象のモデルの集合（学習と集約に使うモデルID）を、毎roundの学習の直前にその時点の割り当てから作り直し、clientの学習に渡す。同じroundの学習と集約の候補には同じ集合を使う。分割したroundの次のroundから、新しいモデルが学習・集約の対象に入り、割り当て直したclientの更新で集約される。対象の集合の通知の件数はこれと別に数える（design.md）。分割の前の学習の件数の記録は付け替えない。
4. The system shall 次の時刻の予測と割り当ての写しには、その時刻の最後のroundの後の割り当てを使う。
5. The system shall 分割の判定の記録`cfl_split_check`に、時刻の中のroundの番号（0から）を持つ列を足す。保存形式の版を上げる。

### Requirement 3: test
#### Acceptance Criteria
1. The validation shall 既定の $R=1$ で、全構成の設定・識別子・基準版1〜16のgoldenが変わらないことを確かめる。
2. The validation shall 掃引の値と構成の制約（2構成だけ許し、ほかを拒否）を確かめる。
3. The validation shall `feddrift-window`の $R=3$ の小さい全体runで、各時刻にroundが3回あり、どのroundも同じ窓の標本で学習し、上りと下りのパラメータの通信の回数が $R=1$ の3倍であることを確かめる。
4. The validation shall CFLのサーバの経路で、時刻の途中（3 roundのうち最初）に分割が決定的に起きる入力を使い、(1) 分割したroundの集約の省略、(2) 残りのroundで割り当て直したclientが新しいモデルをその時刻の窓の標本で学習し、元のモデルの学習の標本にその標本が残らないこと、(3) 新しいモデルが学習・集約の対象に入ること、(4) 記録のroundの番号、(5) 次の時刻の割り当てと予測のモデル、を確かめる。
5. The validation shall CFLの $R=3$ の小さい全体runで、既存の小さい全体runの照合（割り当ての保持、窓、予測、判定の照合、上限、検出0・統合0）が、roundごとの判定と時刻の途中の割り当て直しを含めて成り立つことを確かめる。

### Requirement 4: 文書と予備の実行
#### Acceptance Criteria
1. The documentation shall 設定の参照（再生成）、`docs/experiments/differences-from-feddrift.md`のD11とCFL+Windowの節、構成の一覧に、$R$ の指定と、公式との違い（design.mdの「公式との違い」）を書く。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認をして、予備の実行（design.mdの「予備の実行」）のコマンドをユーザーに渡し、集計（分割の回数・時刻、モデルの数、精度、通信、同じ $R$ のFedAvg+Windowとの差）を示す。本規模へ進むかと $R$ の決め方はユーザーが決める。
