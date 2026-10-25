# S3の符号長評価と候補学習を分ける対照

[研究記録](../../docs/research/fedsda-s3-factorial-20261025.md)の研究用CLI。
製品・既存のS3ツールを変更しない。

| arm | 符号長評価 | 実候補学習 | 既定の実行対象 |
|---|---|---|---|
| warm-matched（W） | adapter＋head | adapter＋head | 保存済みを再利用 |
| head-score-full-train（A） | headだけ | adapter＋head | 新規 |
| full-score-head-train（B） | adapter＋head | headだけ | 新規 |
| output-layer-matched（C） | headだけ | headだけ | 保存済みを再利用 |

全cellでwarm選択時の候補初期値は現在モデル。共有部固定、2学習者・ln2、
初期化し直す枝、採用後の通常学習は維持する。D（製品既定）の初期値は別なので、
因子効果はWからの差、標準の精度差はDからの差として両方を表示する。
交差したA/Bは診断用であり、符号長の代理学習者と実候補の学習範囲が異なる。
結果が良くても直接本実装する手順ではない。

```bash
python -m tools.research_s3_factorial.run \
  --reference results/fedsda-output-layer-learner/20261025_local-mnist4-seeds0-3-n2500 \
  --input-proof results/fedsda-s3-contribution/20261025_mnist4-seeds0-3-n2500/audit_proof.json \
  --output results/fedsda-s3-factorial/<unique-label> \
  --seeds 0 1 2 3 --sample-count 2500 --workers 8
python -m tools.research_s3_factorial.analyze results/fedsda-s3-factorial/<unique-label>
```

内部の事前確認は`--print-plan`。出力先を作らない。保存先の上書きは拒否する。
異なる4seed以上・2500標本以上。既定は新しいA/Bだけを実行し、W/C/Dを再利用する。
`--arms`で端点W/Cの再現も追加でき、その場合は秒以外の元の全保存値との一致を要求する。
A/Bを欠く実行は受け付けない。中断後の再開は未対応。失敗はfailures.jsonへ保存する。

入力proofの親ディレクトリのmetadata.jsonとresults.jsonも必要。
検算元の指標・rawのhash・元参照結果SHAとの結び付きを確認する。
過去の1.5GBの生traceを新規実行先へコピーする必要はない。
各新規runの全入力key集合・画像hash/ラベル/概念の列を、その検算済みの署名へ照合する。
数値実行ソースと参照は実行中に固定し、解析器自身は別のhashとして保存する。

手元のworkers合計は14以下。uraniumでは通常64workersを使えるが、同条件の参照と
入力proof一式も実行先へ用意する。入力hashのCPU処理が加わり、再利用runとは負荷が異なるため、
秒数を方式固有の速度差として扱わない。精度・通信・モデルMAC・準備MAC等は共通runnerで測る。

## 手元4seedの結果

MNIST4・2500標本・間隔500では、Bの精度88.254%、既定比+0.268pp、
上り260.238MB・下り277.942MB・4259.043GMAC・1116.8秒。
4seed中3seedで改善したがseed3は−0.636pp。時間は再利用した既定との速度比較には使わない。
C/S3は既定比−0.006ppで、head評価を加えるB→Cは平均−0.274ppだった。
未使用seedでBの安定性を調べる候補であり、製品採用は未決定。詳細・検算は上記研究記録。
