# LEGACY-018: 共有部を持たないモデルでは、クロス評価の下りの通信量が、同じclientへの同じ候補の送信を、重複して数える

- 発見日: 2026-10-12（構成`independent`の移植で、新旧の通信量の差として現れた）。対象: 旧`federated_drift_experiment/servers/clustering.py::_record_cross_evaluation_model_transfer`（基底の実装）と、`servers/shared_backbone.py`の同名の上書き。固定基準748c3aa。
- 発見spec: fedsda-independent-models。種別: 通信量の数え方が、モデルの構造（サーバのクラス）で揃っていない。状態: 再現済み・未修正。移植では、旧と同じ数え方を維持。

## 再現と観測

クロス評価は、候補モデルiと、評価先モデルjの組ごとに、jを保有するclientへ、iを送って評価させる。

- 共有部を持つ構成のサーバ（`SharedBackboneFedSDANoCachedServer`）は、1回のクロス評価の中で、同じclientへ、共有部を1回、同じ候補の概念固有部を1回だけ、数える。
- 共有部を持たない構成（旧の`independent`）のサーバ（`FedSDANoCachedServer`。基底の実装を使う）は、組（i、j）ごとに、評価するclientの数だけ、iの全パラメータを送った、と数える。clientが、評価先モデルを複数保有していると、同じ候補を、その数だけ送ったことになる。

2026-10-12、Windows基準環境で、`tests/refactoring/test_fedsda_configuration_variants.py`の、構成`fedsda-independent-models`の小さい条件（sine2の7条件）のうち3条件で、重複を省く数え方（新実装の、共有部を持つ構造の数え方を、そのまま使った場合）と、実旧の値が違った。下りのバイト数は、重複を省くと5808・45056・19008、実旧は6160・46464・21472（実旧が、約3〜13%多い）。残りの4条件は、同じ値（clientが、評価先モデルを複数保有するクロス評価が、無い）。

## 影響

- 精度、検出、モデルの中身は、変わらない（通信は、模擬で、数えるだけ）。
- 下りの通信量（モデルの数、パラメータの値の数、バイト数）の指標が、共有部を持たない構成で、多めに出る。提案構成と`independent`の通信量を比べると、数え方の違いが、`independent`に不利な向きに混ざる。

## 新実装の扱い

- 構成`fedsda-independent-models`（共有部が空の構造）では、旧と同じく、重複を省かずに数える（`runtime/model_cross_evaluation.py`）。旧実装との照合を通すため。
- 共有部を持つ構造の数え方は、変えていない。
- 重複を省く数え方へ揃えるかどうかは、未定（揃えると、この構成の下りの通信量が、旧と一致しなくなる。構成どうしの比較としては、揃う）。2026-10-12、ユーザーへ報告済み。
