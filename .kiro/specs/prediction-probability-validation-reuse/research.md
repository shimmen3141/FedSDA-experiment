# Research & Design Decisions

## Summary

- **Feature**: `prediction-probability-validation-reuse`
- **Discovery Scope**: Simple Addition
- **Key Findings**（sine2・client 10・標本500、3回のうち最短）:
  - 変更なし 16.73秒。予測の中の、確率のtensorの検査を、変換のときの1回だけにする 15.68秒。標本1件の処理の、損失の評価での、標本の再検査を省く 16.56秒。両方 15.47秒。結果（精度、通信量）は、どれも同じ。
  - プロファイル（client 10・標本300＝全部で3,000標本）: `_validate_prediction_tensor`が27,000回（標本1件あたり9回）。
  - 上の時間は、sourceを変える前の、検査を差し替える実験の値。実装の後の実測は、tasks.mdのImplementation Notes。

## Design Decisions

### Decision: 「検査済み」を伝える引数を足す

- **Alternatives**: (1)関数の検査をなくす。(2)検査済みの値を表す型を足す。(3)keywordの引数で伝える。(4)予測の全体を、1つの関数にまとめる。
- **Selected Approach**: (3)。
- **Rationale**: 既定の挙動と、関数ごとの検査の契約（既存のtest）を、変えない。変更が小さい。(2)は、型と変換が増える。(4)は、関数の分け方を変える。
- **Trade-offs**: 呼出し側が、誤って真を渡すと、検査されない値が計算へ入る。真を渡すのは、標本1件の予測の、変換の直後の呼出しだけにし、その根拠を、コメントに書く。

### Decision: 効果の小さい候補は、採らない

- 標本1件の処理の、損失の評価での再検査は、約1%。その検査は、保持へ入る標本の、入口の検査の一部でもある（training-batch-sampling-validation-costの設計が、根拠にしている）ので、変えない。
