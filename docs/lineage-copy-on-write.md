# Lineage-aware copy-on-write

`NEW_MODEL_CREATION_POLICY=forward_lineage` は、forward検証で採用された候補を
常に新規モデルIDへ登録せず、初期化元の親モデルの利用状況に応じて
`update` または `fork` として確定するFedSDAの実験的な作成方針である。

## クライアント側

候補の学習・検証・採用条件は `forward_persistent` と同じである。

1. 警報直後に既存モデルの再適合を確認する。
2. 候補を検知区間で学習する。
3. 警報後の新着 `K=NEW_MODEL_FORWARD_VALIDATION_SAMPLES` 件で評価する。
4. 候補が最良参照モデルより、独立した前半・後半の双方で良い場合だけ採用する。
5. 採用候補へ `NEW_MODEL_INITIALIZATION` で選ばれた初期化元モデルIDを親として付ける。

`average` 初期化には単一の親がないため、サーバでは常にforkとして扱う。

## サーバ側

候補の回収ラウンドで、親モデルを現在利用する他クライアントと、同じ親を持つ
同時候補を確認する。

- 親を利用する他クライアントがなく、同じ親の候補も1個だけ: 親IDを `update`
- 親を他クライアントが利用中: 旧状態を保存するため新規IDへ `fork`
- 同じ親の候補が同時に複数ある: 上書き競合を避けるため各候補を `fork`
- 親が未登録、一時ID、または単一の親を持たない: `fork`

`update` では親の旧パラメータと通常FedAvgで混合せず、候補パラメータを親IDへ
直接反映してから配布する。これにより、単なる局所適応でグローバルモデルIDを
増やすことを避ける。

## 診断値

Pareto CSVには次を保存する。

- `lineage_fork_count`
- `lineage_update_count`

raw NPZには各判定のラウンド、クライアントID、親ID、確定先ID、`fork/update`を
`copy_on_write_*` 配列として保存する。

## 注意

この方式が判定するのは「旧親モデルを別IDとして保存する必要があるか」であり、
ドリフト警報そのものの正しさではない。警報precision、新規候補のforward採否、
copy-on-write判定は分けて評価する必要がある。
