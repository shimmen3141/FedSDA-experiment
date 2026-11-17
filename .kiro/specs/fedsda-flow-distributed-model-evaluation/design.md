# 設計

以下の初回設計にあるdistribution・clustering・receive-models・cluster-distributeの表示は、末尾「サーバ内の整理」と要求8〜10で置換済み。最新の処理箱・線名は末尾を正とする。

- 既存R05を全モデル先行配布に変更。R05a（評価依頼）のみ追加しbuildのR05直後へ挿入。R06は推論比較と結果返信を維持。R07統合、R08必要時の変更分再送、R09受信内容の反映へ変更。R10/R11は保持。
- 既存cross-evaluation枠を「配布されたモデル群」「反映待ち」と改称しR05/R05a/R06/R07/R08/R09で保持。これらのコマは旧配布時snapshotの図を隠して上に受信枠を重ねる。実際のsnapshotを上書きした意味にはしない。受信枠が非活性の他コマは従来の薄表示を保つ。比較矢印はR06のみ。右側の対象全体を囲む点線枠はR06とR09に表示し、反映矢印の終点もその左辺とする。
- receive-models: distribution→受信枠。R05は全モデル、R08は変更分のみ。経路は290,360→315,360→315,225→435,225。既存evaluation-requestはclustering→受信枠の軽量な評価依頼へ改称しR05aだけ。両線を同時活性化しない。
- R08でcluster-distributeとreceive-modelsを表示し、統合時のみという条件をcaptionに記す。共有部は変わった場合も対象と短く明記。受信枠はR07でも残し評価後破棄をしない。
- apply-received: 受信枠右辺805,365→更新用モデル群左辺885,365のクライアント内部線。R09の旧distribute線と入れ替え、サーバからの全モデル転送を示さない。ラベルは反映。
- 先行配布・変更分再送ラベルと受信枠の補助文はframeに対応して更新する。初期DOMは先行配布の名称。モデル群の見出しも受信モデル群と手元更新用モデル群に合わせる。R09の受信枠補助文は「反映待ち」でなく反映中の表示にする。
- サーバ共有以外のframe契約と操作handlerは不変を比較する。3画面×全6分岐を走査しR05〜R09の順序、限定表示/接続/説明/色/濃淡、R10コピーを確認。代表PNGを目視し、線とラベルの干渉を確認する。

## 追記：サーバ内の整理
- registration 40,115,250,48、aggregation 40,175,250,48、synchronization-conditionは中心165,251・幅230高さ38のひし形、server-evaluation 40,287,250,48、consolidation 40,352,250,48。サーバ外枠は維持。旧clustering/distribution箱を評価と統合に置換する。
- server-orderの薄い静的な4矢印で、登録→集約→条件→評価→統合の条件成立時の順序を示す。actor/edge活性リストには属さず、通信の移動アニメーションを使わない。
- early-models: aggregation右辺290,195→315,195→315,225→435,225。changed-models: consolidation右辺290,372→315,372→315,225→435,225。ラベルは先行配布/全モデルと変更分再送/統合時のみ。R05/R08だけ活性。
- evaluation-requestはserver-evaluation右辺290,301から受信枠へ。結果返信は同じ処理の右辺290,320へ。upload終点はaggregation右辺290,195へ追随。
- R04はconditionのみ、R05はaggregation+受信枠、R05a/R06はserver-evaluation+対応client要素、R07はconsolidation+受信枠、R08はconsolidation+受信枠。旧aggregate-cluster/cluster-distribute/receive-modelsを除去し、他frame契約と操作は保持。

- サーバ箱の見出しは18px。画面高による文字bboxの差を含め余白を確保するため、箱高さを48pxにし、登録の2行baselineは137/155、集約197/215、評価309/328、統合374/392とする。ポートと順序ガイドの端点は維持する。
