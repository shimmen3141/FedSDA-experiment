# 候補検証の待機と選別を切り分ける診断

研究目標は検証あり/なし両方の精度・通信前線を上回ること。[サーバ受領結果](../research/fedsda-server-results-20261018.md)と[単純化の分析](../research/fedsda-candidate-validation-simplification-20261018.md)に基づく原因切り分け用の対照であり、新しい既定手法ではない。

|mode|内容|
|---|---|
|validation（V）|現行の検証あり|
|immediate（O）|現行の検証なし。候補作成前の既存モデル再利用は維持|
|delayed-forced（D）|Vと同じ10標本を待ち、元の評価を計算後に必ず候補採用。未完了候補は現行どおり終端回収|

Dの記録理由は`diagnostic_forced_adoption_after_validation_wait`。通常合格と区別し、元の採否・比較損失・参照もJSONのintervention_eventsへ保持する。
D−Oは待機だけでなく候補凍結・学習先・参照作成などを含み、D−Vも方策全体の差である。介入履歴の元判定はD軌道上での判定であって、別のV軌道の反実仮想の最終精度ではない。

## 実行

Linuxの新実装checkoutルート、tmux内で実行する。単独の実験は最大14workers、複数セッションの合計14以下。本例は7workerで、別の7worker実験1つと並行できる。既に7+7workerが稼働中なら、どちらかの完了後に開始する。単独なら14workerへ変更してよい。

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python run_candidate_validation_waiting_diagnostic.py \
  --cases mnist2:100 mnist2:200 sea4:50 mnist4:500 circle2:200 sine2:500 \
  --seeds 0 1 2 --modes validation immediate delayed-forced \
  --sample-count 5000 --workers 7 \
  --output-directory results/fedsda-candidate-validation/waiting-diagnostic-n5000-seeds0-2 \
  --purpose '候補検証の待機・凍結と選別を分け、検証なしの利得を保ちつつ低下を抑える制御の必要性を6条件で診断する'
```

6条件×3seed×3方式＝54run。各client5000標本、10client、FIFO30・共通T120。MNIST2・100/200とSEA4・50は低下側、MNIST4・500とCIRCLE2・200は利得側、SINE2・500はseed依存の補助条件。
初期結果に応じてseed5〜7も同じ条件で追加する場合は、`--seeds 5 6 7`と別の出力先`waiting-diagnostic-n5000-seeds5-7`へ変更し54run追加する。異なる環境のV/O/Dを片側ずつ混ぜない。
この文書の54runは計画であり、実行済みの主張ではない。実際に開始した保存先・来歴は台帳とresumeへ記録する。

## 保存と失敗時

各runを専用ディレクトリへ原子的に予約し、records.npzとresult.jsonを保存する。result.jsonのformatは`candidate-validation-waiting-diagnostic-v1`であり、通常のformat2 JSONとして既存解析CLIへ渡さない。
完全設定、元sourceとscriptのhash、commit/dirty、環境、実指標、NPZ hash、外側RNGの前後hashと介入内容を保存する。Python/NumPy/torchのRNG fingerprintは同一環境でのみ比較する。

同名runが存在すれば空でも拒否する。中断・失敗時の予約も保持し、既存ファイルを上書きしない。この診断CLIは自動再開を持たない。部分完了した実験では完了結果を保持し、新しい出力先で必要case/seed/modeだけを指定する。既存結果を削除してやり直さない。

## 判断

- DがOの低下を抑え、Vに近づくなら、待機中の扱いが有効な可能性を調べる。
- DでもVより悪ければ、待機だけでは足りず品質選別が必要な可能性がある。
- DがOの利得を失うなら、凍結したまま待つ対策を一律に使わない。
- 次の共有ゲートGは未実装。即時ローカル学習と共有登録を分ける案として、Dの結果後に判断する。

精度だけでなく上下りbyte、合計、推定MAC、保有モデル数、採否と概念別精度を見る。通信は記録パラメータbyteであり、実ネットワークの全byteではない。許容低下幅は未設定、自動採否なし。seed0〜9は探索用であり最終評価用とは扱わない。

手元の1000標本・18runの接続確認は[結果記録](../research/fedsda-candidate-waiting-diagnostic-20261018.md)を参照。性能採否用の54runとは区別する。
