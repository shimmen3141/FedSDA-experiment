# MNIST4の限定W対照（新規20run）

[研究計画・検証記録](../../docs/research/fedsda-s3-warm-control-20261025.md)。
5000標本の本規模で効果が逆になった間隔100/500に限定し、Wだけを追加する。
Wは現在モデルから初期化し、adapter＋出力層を学習する。Bは同じ初期値から出力層だけを学ぶ。
既存D/B40件を再利用し、W−DとB−Wを分ける。製品・既存研究実行コードは変更しない。

## uraniumで実行

`refactor/architecture`でtmux内から実行する。参照は前回の本規模結果一式。
直下に結果がある形式と、直下の唯一の完了済み日時サブディレクトリの両方を受け付ける。
複数の完了結果がある場合は、`--reference`を具体的な保存先へ指定する。

```bash
git pull --ff-only origin refactor/architecture
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -u -m tools.research_s3_warm_control.run \
  --reference results/fedsda-s3-fullscale \
  --output results/fedsda-s3-warm-control/20261025_uranium-mnist4-i100-500-seeds0-9-n5000 \
  --intervals 100 500 --seeds 0 1 2 3 4 5 6 7 8 9 --workers 64

python -m tools.research_s3_warm_control.analyze \
  results/fedsda-s3-warm-control/20261025_uranium-mnist4-i100-500-seeds0-9-n5000 \
  --reference results/fedsda-s3-fullscale
```

MNIST4・各client5000標本・10clientsは固定。新規W20runで、同時processは最大20。
`START W=0/20 reused_D_B=40`、完了ごとに`W 7/20 ...`を表示する。
中断後は同じ実行コマンドに`--resume`を追加。完了済みを飛ばす。
同じ出力先を同時実行せず、実行中に数値ソースを変更しない。
内部の事前確認は読み取り専用`--print-plan`。出力先は作らない。
Windowsからのdry planは環境検査を実行時へ延期し、実行時は本規模D/Bと同じ環境を必須とする。

## 何を検証するか

- D/Bの全設定・元既定の来歴・環境・source/result/checkpoint hashを再確認する。
- Wの全入力画像/ラベル/概念の署名を、同じ条件のBの保存署名へ直接照合する。
- 精度・既定差pp・W差pp・上り/下りbytes・モデルMAC・秒を、条件/seed別に保存する。
- seedごとに`B−D = (W−D) + (B−W)`を検算する。
- 旧既定の時間N/Aや入力画像の直接照合不能は引き継ぐ。W/Bの入力一致と混同しない。
- 同時負荷・計測処理が異なるので、秒を方式固有の速度差とは扱わない。

`runs/*.json`に完了run、`predictions/*.json`に全予測の入力対応/正誤を保存。
予測の戻り値を読むだけで追加NN推論は0。全50000標本のkey集合・重複・正誤・入力署名・製品の正誤digestを照合し、
再開/解析時も系列hashを再確認する。結果をコピーするときはpredictionsも含める。
参照を移動した場合、解析時の`--reference`で新しい場所を指定でき、内容SHAの一致を必須とする。

## 手元の累積計測

同じMNIST4入力の2予測系列がある場合だけ、500標本ごとの区間差・累積差を出せる。
クライアントごとのsample indexを揃え、全10clientsの被覆と画像/ラベル/概念の一致を確認する。
明示したsample-countの末尾が両方欠けている場合も拒否する。旧10クラス確率形式にも対応。

```bash
python -m tools.research_s3_warm_control.cumulative \
  --baseline-trace <Dの予測系列JSON> --candidate-trace <BまたはWの予測系列JSON> \
  --sample-count 5000 --window-size 500 --output <未使用の解析結果JSON>
```

到着結果の確認で、Dの元NPZに正誤/ラベル/概念系列が残っていることを確認した。D/B両方がないとした準備時の説明を訂正する。
Dは画像hashが無いので、この厳格JSON対照CLIへWの画像hashを転記して使わない。
W/Dの累積差は、別の検算scriptでkey/ラベル/概念を対応させて追加実験なしで復元済み。
Bの系列だけが未保存なので、B/W累積差に必要な追加再実行はB側だけとなる。
手元では既存2500標本seed3のB/D系列で解析を検証する。これを5000標本の原因分析とは呼ばない。

## 到着結果

[限定W結果](../../docs/research/fedsda-s3-warm-control-results-20261025.md)。W20run完了・失敗0、Luna監査PASS。W−Dは両間隔で約+0.024pp、B−Wはi100負/i500正で、主に学習範囲側に条件依存が出た。本実装は引き続き見送り、次の限定診断は結果記録を参照。
