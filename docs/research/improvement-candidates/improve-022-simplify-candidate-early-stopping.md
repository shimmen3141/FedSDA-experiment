# IMPROVE-022: 候補の学習の早期終了を単純にする（共有部を固定した後）

- 日付: 2026-10-20
- 対象: 候補の学習（`learning/training/candidate_epoch_training.py`の`validation_loss_early_stopping`）、最終構成の定数（`_CANDIDATE_MAXIMUM_EPOCH_COUNT = 30`、`_CANDIDATE_VALIDATION_SAMPLE_FRACTION = 0.2`、`_CANDIDATE_CONSECUTIVE_NON_IMPROVING_EPOCH_LIMIT = 3`、`_MINIMUM_CANDIDATE_MEAN_LOSS_IMPROVEMENT = 0.0001`）
- 基準commit: `053fefc`（基準版7）
- 種別: 簡略化（数値・乱数が変わる変更）
- 状態: 未検証・未採用
- 優先: 4案の2番目（研究ツールに変種を1つ足すだけで試せ、壊すものが少ない）

## 前提（なぜ取り組むか）

2026-10-20から、候補の学習では共有部を固定し、概念固有部（rank 8のadapterと出力層）だけを学習する。学習するパラメータは、共有部も学習していたときより、ずっと少ない。それでも候補の学習は、警報の区間（5〜30標本程度）からさらに20%を検証用に取り分け、最大30エポック、改善が3エポック続かなければ止める早期終了を行い、最良のエポックのパラメータへ戻している。

区間が短いと、検証用の標本は1〜6個しかなく、早期終了の判定は雑音に近い。学習用の標本も2割減る。固定のエポック数で足りるなら、学習の手順と乱数の使い方（検証用の分割の`randperm`）が単純になり、使う標本も増える。

## コードで確認した事実

- `train_candidate_classifier_epochs`は、`candidate_training_strategy`が`validation_loss_early_stopping`のとき、区間を乱数で分け（検証用は`max(1, round(標本数×0.2))`）、1エポックごとに検証用の平均損失を測り、改善（`minimum_validation_loss_decrease`より大きい減少）が3回続けて無ければ止め、最良のパラメータを読み込む。学習用の標本が1未満なら、全標本で固定エポックの学習をする。
- 選択肢には、`fixed_epoch_training`（全標本で`maximum_epoch_count`エポック）と`skip_training`がすでにある。新しい処理は要らず、設定の値を替えるだけで試せる。
- 最終構成の検査に、「候補の改善の最小幅（`minimum_candidate_mean_loss_improvement`）を、早期終了の最小幅と同じ値にする」制約がある（`FedsdaRunClientSettings`）。早期終了を使わなくても、この制約は残る（IMPROVE-023で整理する）。

## 方針

1. 研究ツール（`tools/research_provisional_candidate_expert/run.py`）に、既定（`fedsda`）の候補の学習を`fixed_epoch_training`へ替えた変種を足す。エポック数は数通り（例: 5・10・30）。設定の置換だけで作れる（関数の差し替えは要らない）。
2. 手元（2500標本、seed 0〜2、間隔50・200ほか）で、精度と、候補の学習の計算量（`model_computation_counts`）を比べる。
3. 効果が中立以上なら、本規模（uranium）で確かめる。

## 懸念

- 固定エポックで過学習すると、新しいモデルの初期の予測が悪くなる。共有部を固定したので、過学習の影響は概念固有部に限られるが、確かめるまでは分からない。
- 早期終了は「最良のエポックへ戻す」ので、学習しすぎを防いでいた可能性がある。エポック数の選び方が新しい調整項目になるなら、単純化の利点は減る。

## 必要な比較と採否の基準

- 比べる相手: 現在の既定（`fedsda`）。
- 指標: 精度（dataset×間隔ごと）、新しいモデルの採用直後の精度（回復期の精度）、候補の学習の計算量。
- 採否: 精度を落とさずに単純になるなら採用を検討する。

## 検証結果

（未記入）
