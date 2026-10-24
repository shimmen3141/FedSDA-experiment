# S3の候補寄与診断

製品・既存のS3研究ツールを変更せず、3構成のMDL判断、全予測、全新候補の来歴、
統合前の混合から候補を除いた差を記録する。詳細は
[診断記録](../../docs/research/fedsda-s3-contribution-diagnostic-20261025.md)を参照。

```bash
python -m tools.research_s3_contribution.run \
  --reference results/fedsda-output-layer-learner/20261025_local-mnist4-seeds0-3-n2500 \
  --output results/fedsda-s3-contribution/<unique-label> \
  --seeds 0 1 2 3 --sample-count 2500 --workers 8
python -m tools.research_s3_contribution.analyze \
  results/fedsda-s3-contribution/<unique-label>
```

実行前の内部確認には`--print-plan`を足す。出力先は作らない。
MNIST4、集約間隔500、既定/初期値整合対照/S3の3構成を実行する。
参照は元S3 CLIの同条件の結果で、必須。参照の完了・数値実行ソース・条件を確認する。
4seed以上・2500標本以上に限定し、既存出力先への上書きは拒否する。再開機能はない。
手元のworkers合計は14以下、uraniumは通常64。ただし参照成果物も実行先へ用意する。

全runで秒以外の参照保存値を完全照合する。モデルMAC、精度/パラメータdigest、通信、
元S3イベントも含む。全予測の正誤digestも元の値に照合する。
`complete.json`は実行ソースの照合完了を示す。全標本の検証済みという意味ではない。
必ず`analyze`で10clients×全sample indexの正確なkey集合、確率からの正誤、
構成間の入力対応、MDLの再現が成功し、`summary.json`を生成してから結果を採用する。
計測は既存の同期側/局所側の確率を使い、追加のNN呼出しはしない。
確率の再結合・hash・JSON保存等の処理は増えるため、秒数は方式固有の速度と扱わない。
共通runnerの秒数と、保存を含む診断全体の秒数は別に残す。

候補の単体除去差は複数候補間で足せない。warm枝の未統合候補を同時に除く指標を別に持つ。
残余の重みがない標本は欠測、統合後の来歴は元候補の寄与率ではない。
採用しなかった世界を再学習する介入ではなく、同じ時点のモデル・重みを固定した局所寄与である。

この4seedは既に結果を見ている探索用。採用時特徴と事後ラベルを混同せず、
規則を作る場合は固定後に新しいseedで検証する。
