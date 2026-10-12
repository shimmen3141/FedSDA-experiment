# Implementation Plan

- [x] 1. クロス評価の下り転送を構造によらず呼出し内で重複除外する
  - 複数候補・複数client・複数保有、評価先無し、client抽出、再呼出しの再現testを先に書きREDを確認する。
  - 既存集合を共用して修正し、新規testと共有部ありの既存部品testでGREENを確認する。不正入力の更新前拒否と途中失敗の既存部品testも実行する。
  - _Requirements: 1, 2_

- [x] 2. 修正済みの通信量を全体run・成果物まで厳密に照合する
  - 実旧の元処理を維持したイベント観測から重複分を計算し、独立構成だけの期待値へ対応付ける。
  - 合成データの重複あり/なしとMNIST全状態照合、保存結果の通信量を確認する。
  - 全task後に独立CLI実装レビューと指摘反映、全pytest・品質・fresh・Linuxの対象/既存goldenを通し、固定旧/goldenを変えていないことを確認する。
  - _Requirements: 2, 3_

## Implementation Notes
- 仕様独立Luna medium: 初回Major1（同値別IDの識別）とMinor1（途中失敗のtask記載）を反映。重大範囲再確認PASS。旧メソッド名の明記と既存全状態照合helperの配置追記は文面・配置補足で契約変更なし。
- 全構成golden、他LEGACY、旧実装削除は後続。変異testは既定で未実施。

- task1 RED: 初回fixtureの必須統計不足は訂正しREDに数えない。その後1failed/11passed、重複ケースの下り3項目のみ不一致（240 vs 144byte）。修正後部品/既存クロス評価33passed。
- task2 保存testの型名メタデータ不足を期待値に反映し、新規14passed。当該時点では構成全体対照とfreshが進行中だった。その後のレビューと最終検証の結果は以下。

- 独立Haiku 5.5 medium実装レビューPASS（Blocker/Majorなし）。Minor1/2: 完了時のfinding/README/承認文面更新を反映。Minor3: helper内の実旧補正照合と単独API保存の二段階をtest docstringで明記。Minor4: 観測wrapperを旧payloadと同じTensor限定に補足。Minor5: 空共有部でも重複除外するコメントを追加。任意6: 経路の観測条件は既存小規模の条件集合で有無を確かめる目的のため維持。Minor以下につき再レビュー不要。
- 構成一括の初回は102passed/2failed。保存形式の期待値不足は新規14testで修正確認済み。クロス評価を全く実行しない条件では観測属性が作られないため、経路集計だけを0既定値に修正した（旧比較の期待値helperは元から0既定）。

- 最終検証: 全pytest 13660passed/8skipped/9warnings、696.88秒、exit0。前spec13646passedに新規14件を足した数と一致。JUnit13668case、新規14と旧test_regression/test_proposed_regression各1case成功。旧11/最終3goldenはそれぞれそのcase内で検証。JUnit: ../../venv/refactoring-tests/cross-evaluation-download-deduplication-full.xml。
- 最終品質: Ruff check/format336files、Pyright0、pip check成功、names1source登録漏れなし、progress2/2。fresh直接exit0/16flows、独立構成の新実装だけのrun/保存と旧/test import不在を確認。
- Linux: WSL Ubuntu固定環境の新規14とFedDrift既存golden4は18passed/24.09秒。レビューMinor補足後の新規14は14passed/42.01秒。最初の選択式ではgoldenが選ばれなかったため、FedSDA環境別goldenをnode IDで明示し3passed/76.50秒。Windows/Linuxのgoldenは更新していない。
- 受入れ判定: 要求1→呼出しごとの候補/client重複除外と全組テスト、要求2→抽出/RNG/評価/メッセージ維持と既存拒否/途中失敗/全状態対照、要求3→実旧イベント観測による補正期待値と合成/MNIST/保存/全回帰。3要求領域を満たし、依存方向/API/保存形式の変更なし、未完task/blockedなし。kiro-validate-impl/verify-completion: GO、証拠に対応する範囲でVERIFIED。
- 未実施: 独立レビュー担当による全pytest/Linux/構成一括対照の再現、変異test。全構成goldenの作成と旧実装削除はこのspecの対象外。

- 検証対象source commit: `48bcf8b655eee6cb48a2d9e4df5610e50aafe0d4`。tracked Python＋固定2golden計440パス、LF sha256 `a66c9450f28cb582481078ad3a91dd4e9306949b08e13cb384ef2617b50b4fc4`。identityで未commit差分なし、固定748c3aaとの差分なし、JUnit/旧golden case/件数の一致を確認。完了記録のcommitでPython/goldenは変更しない。
