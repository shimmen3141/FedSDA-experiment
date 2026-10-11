# Research & Design Decisions

- 旧の`ESRFedSDAClient`: 検出器1つ（`BoundedMeanEDetector`）。e値は、検出器の`log_e_value`。警報は、検出器の`drift_detected`。新しい概念の区間の長さの推定は、検出器の`width`。計算の計数は、更新1回と、検出器の有効な候補の数。始め直しの基準は、現在のモデルの損失統計の平均（提案構成と同じ）。
- 旧の`ClassConditionalESRFedSDAClient`は、これを継承して、クラス別の検出器と、混合を足したもの。新の監視の、全体の検出器（`BoundedLossESRDetector`）は、旧の検出器1つと、すでに照合してある。
- 値の名前は、既存の値（`overall_and_true_class_losses`）に合わせる: `overall_loss`。
