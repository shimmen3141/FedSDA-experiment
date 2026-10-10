# Research & Design Decisions

## Summary

- **Feature**: `environment-keyed-regression-goldens`
- **Discovery Scope**: Extension（testと基準のファイルの置き方を変える。sourceは変えない）
- **Key Findings**:
  - 研究室サーバとWSLは、どちらもLinux・x86_64・torch 2.12.1+cpuだが、PythonとNumPyの版が違い、旧実装のsine2の結果が違う（候補の棄却が2件と3件）。OSの名前では、goldenを選べない。
  - 既存の回帰testの`environment()`は、OS・機種・Python・NumPy・torchの版・device・dtype・thread数を返し、goldenの`_env`に、同じ形で記録されている。これを、goldenを選ぶ鍵に使える。

## Design Decisions

### Decision: goldenを選ぶ鍵は、環境の記録の完全な一致

- **Alternatives Considered**: (1)OSの名前——サーバとWSLを区別できない。(2)環境変数や設定ファイルで、goldenを指定する——cloneした人に、設定を求める。(3)近い版のgoldenを使う——合わないときに、失敗の理由が分かりにくい。
- **Selected Approach**: 実行環境の`environment()`と、goldenの`_env`が、完全に一致するものを選ぶ。なければ、goldenとの照合だけをskipする。
- **Rationale**: 設定が要らない。合わない環境で、誤った失敗が出ない。
- **Trade-off**: 版が1つ変わるだけで、goldenが合わなくなり、照合がskipになる（黙って照合が減る）。開発の環境（Windowsの基準環境とWSL）では、skipの件数を、検証の記録で確かめる。Windowsの基準環境では、既存の回帰testが、版が違っても、警告して比較する（変えない）。

### Decision: ふだんのspecで、研究室サーバでの実行を求めない

- **Selected Approach**: 開発中の照合は、主担当が、Windowsの基準環境とWSLで行う。研究室サーバ用のgoldenは、ユーザーが1回作れば、その後は、`git pull`の後に、照合したいときだけ実行すればよい。
- **Rationale**: 新実装と旧実装の一致は、同じprocessの中の照合で、どの環境でも確かめられる。goldenを作り直す必要があるのは、意図したアルゴリズムの変更を承認して、結果が変わるときだけ。

### Decision: 固定のWindows用のgoldenは、動かさずに、選択の対象に含める

- **Selected Approach**: 選択の候補は、固定のWindows用のgoldenと、新しいディレクトリのファイル。
- **Rationale**: 固定の範囲を変えない。選び方は、1つ。

## Risks & Mitigations

- 同じ環境の記録で、結果が違う計算機がある（CPUの違いなど）— その計算機では、goldenとの照合が失敗する。環境の記録に、区別する項目を足すことを、そのときに検討する。
- 版が変わって、照合が黙ってskipになる — 検証の記録で、skipの件数と理由を確かめる。
