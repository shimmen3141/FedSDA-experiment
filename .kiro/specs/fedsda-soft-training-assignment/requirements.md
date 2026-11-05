# Requirements Document

## Project Description (Input)
- 誰が困っているか: FedHASP（構成`fedsda`）を論文にまとめる研究者（ユーザー）。
- 今の状況: 構成要素の分析は、固い割り当ての細かい要素を1つずつ外すものだけで、固い割り当て（各標本を1つのモデルだけの学習に使う）そのものを、ほかの割り当て方と比べていない。既存の診断（`docs/research/fedhasp-oracle-assignment-v16.md`）で、固い割り当てを真のコンセプトに合わせた割り当てに置き換えると+0.41pp（間隔50・500、seed 0〜9）と分かっている。
- 2026-11-05のユーザー判断: 「多様なモデルに分けるために、今の固い割り当てが有効である」ことを、固い学習を柔らかい学習（各標本で全モデルを、その標本への責任度で重み付けて学習する。混合モデルのEMやmixture of expertsの標準的な分け方）に置き換えた構成と比べて示す。予測（全モデルを混ぜる）は変えない。規模は10データ×間隔50・500×seed 0〜9（200 run）、余裕があれば間隔100・200を足す。実装はGPT-6.1-sol。
- 何を変えるか: 構成`fedsda`の変種`fedsda-soft-training`を足す。学習の割り当て以外（予測、警報、帰属の保留、MDLによる変化点と帰属先の選択、新しいモデルの作成と候補の学習、統合、集約の手順）は`fedsda`と同じにする。

## Boundary Context
- 対象: 新しい構成1つ、その学習の経路、多様さの指標（学習データのコンセプトの純度）の記録と導出、test、文書、実行のコマンド。
- 対象外: `fedsda`とほかの全構成の設定・値・runの識別子・golden、FedDrift系の構成。予測の仕組みは変えない。

## Requirements
### Requirement 1: 柔らかい学習の構成
#### Acceptance Criteria
1. The system shall 構成`fedsda-soft-training`と`fedsda-training-mass-diagnostics`を持つ。前者は学習の割り当て以外の設定と処理を`fedsda`と同じにし、後者は`fedsda`に学習の量のコンセプト別の記録だけを足す（挙動は`fedsda`と同じ）。
2. The system shall 帰属の保留（FIFO）から標本を放す時点（`fedsda`で標本をモデルのデータストアへ入れる時点）で、その標本を、モデルごとのデータストアではなく、clientの1つの共有の学習ストアへ入れ、その時点の保有モデルの予測の重みと局所のパラメータでの真のラベルの予測確率から求めた責任度 $q_{i,m}$ を一緒に蓄え、以後は更新しない。定義・統合と保有をやめたモデルでの扱いはdesign.mdの「責任度」。
3. The system shall 学習の各更新で、`fedsda`と同じ回数・同じ時点・同じ件数 $B$・同じoptimizerと学習率・同じ共同更新の式で、design.mdの参加の条件を満たす各保有モデルについて、共有の学習ストアから確率 $q_{i,m}/\sum_j q_{j,m}$ で $B$ 件を復元抽出したミニバッチの平均の損失（責任度で正規化した重み付きの平均の損失の不偏な推定）で、特徴抽出器と各モデルの概念固有部を同時に更新する。
4. The system shall サーバの集約で、各モデルの概念固有部のclientの重みを、`fedsda`の「そのclientでそのモデルに帰属した標本の数」の代わりに柔らかい件数 $n_m=\sum_i q_{i,m}$ にし、集約に参加するモデルの条件（`fedsda`は件数が0より大きい）と特徴抽出器のclientの重み（`fedsda`は参加するモデルの件数の和）も、件数を $n_m$ に置き換えた同じ規則にする。
5. The system shall 警報・MDLによる変化点と帰属先の選択・新しいモデルの作成・候補の学習・統合の判断を`fedsda`と同じにする。帰属先の選択の結果（今のモデル）は、警報の監視と予測の仕組みでは`fedsda`と同じに使い、学習には使わない（学習は全モデルの柔らかい学習）。
6. The system shall 既存の全部の構成の名前・順・保存設定・runの識別子・goldenを変えない。

### Requirement 2: 多様さの指標
#### Acceptance Criteria
1. The system shall 学習の量のコンセプト別の記録を、記録だけを足す設定の項目（既定は記録しない）として持ち、`fedsda-soft-training`と、`fedsda`に記録だけを足した構成`fedsda-training-mass-diagnostics`で記録する。純度（design.md）を両方で同じ定義で求められるようにする。`fedsda`とほかの既存の構成の出力・設定・識別子は変えない。

### Requirement 3: test
#### Acceptance Criteria
1. The validation shall 責任度の値（分母0の場合を含む）、統合と保有をやめたモデルでの列の扱い、柔らかい件数と参加の条件、$q$ に比例する抽出、集約の重みが、design.mdの定義と一致することを、固定の入力で確かめる。
2. The validation shall 小さい全体runで、学習以外の処理（予測、警報、選択、統合の判断の入口）が`fedsda`と同じ経路を通ること、全保有モデルが各更新で学習されること、共有の学習ストアの件数と $q$ の記録、純度の導出を確かめる。
3. The validation shall $q$ をone-hotにしたとき、同じミニバッチを差し込めば`fedsda`と損失・更新の後のパラメータが一致すること（更新の一致）と、抜いた標本がそのモデルに帰属した標本だけで重複が無いこと（抜き方）を、分けて確かめる（design.mdの「固い場合との一致」）。
4. The validation shall 基準版1〜16のgoldenの照合と、既存の構成・掃引・識別子・依存境界のtest、`tests/test_regression.py`、`tests/test_proposed_regression.py`が値を変えずに通ることを確かめる。

### Requirement 4: 文書と実行
#### Acceptance Criteria
1. The documentation shall 構成の一覧に`fedsda-soft-training`の目的と、`fedsda`との違いを書き、設定参照を再生成する。
2. The experiment shall 実装・commit・pushの後、主担当が内部確認をして、本規模の実行（2構成×10データ×間隔50・500×seed 0〜9、400 run）のコマンドをユーザーに渡し、2構成を同じseedで対にして集計し、`fedsda-training-mass-diagnostics`の精度が`fedsda`の既存の結果と一致することを確かめる。
