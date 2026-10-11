# Research & Design Decisions

- 旧の`hard-routing`のclientは、結合のmixinを持たない。`_record_prediction`は、基底のもの（現在のモデルの`predict`、計算の計数は、予測1回）。`routing_diagnostics`ほかの属性が無く、旧の指標の計算は、`getattr`の既定（空）から、内訳の指標を0にする。保存する離散列も、結合の3列が無い。
- 旧のサーバは、再較正の設定が`none`でなければ、clientの`recalibrate_routing_after_aggregation`を呼ぶ。このclientは、重みを持たないので、呼ぶと失敗する。旧の`hard-routing`の指定は、設定を渡さず、既定の`none`になる。
- 損失の監視に使う損失は、予測とは別に、現在のモデルで評価する（提案構成と同じ。旧も同じ）。
- 最初の実装で、指標と計算量は一致した。直したのは、(1)結合して予測した標本の数（新は、予測した標本の全部を数えていた）、(2)使われないAdaHedgeの証拠が、学習帰属の変更で、始め直しの回数を数えていたこと、(3)照合の土台（実旧に無い列と指標）。
