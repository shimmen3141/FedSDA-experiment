# Research / Decisions

seed引数の定義はsrcの通常sweep、挙動比較script、予測初期質量research run、component ablation analyzerの4か所。ほかの研究実行入口は通常sweepを委譲利用している。

argparseの独自Actionで複数tokenの正規化を1か所へ集める。[Python公式Action仕様](https://docs.python.org/3/library/argparse.html#action-classes)を確認。既存のnargsと列挙を維持し、カンマ/刻み幅の別構文は増やさない。逆順・重複は旧の重複拒否方針に合わせる。0-4294967295のような範囲が巨大なlistを確保しないよう、展開前に個数を検査する。

他軸は提案に留める。集約間隔など等間隔の整数軸は刻み幅指定の候補、dataset/構成は明示的なgroupの候補。floatの範囲は丸めと終端の曖昧さがあるため、今回一括で導入しない。

適用skillはfable-methodとcc-sddのinit/requirements/design/tasks/impl/validate-impl。既存境界と正規化後の不変条件を優先する。新依存なし。
