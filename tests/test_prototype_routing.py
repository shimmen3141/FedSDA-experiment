import torch

from federated_drift_experiment.diagnostics import PrototypeRoutingDiagnostics


def _scores(value):
    return torch.tensor([[value]], dtype=torch.float32)


def test_prototype_routing_uses_only_past_winning_region():
    diagnostics = PrototypeRoutingDiagnostics()
    predicted_classes = {0: 0, 1: 1}
    prediction_scores = {0: _scores(0.1), 1: _scores(0.9)}
    model_losses = {0: 0.1, 1: 0.9}
    target = torch.tensor([[0.0]])

    # 初回はprototypeがなく、誤答するfallback mixtureをそのまま使う。
    first = diagnostics.observe(
        features=torch.tensor([[1.0, 0.0]]),
        predicted_classes=predicted_classes,
        prediction_scores=prediction_scores,
        model_losses=model_losses,
        fallback_scores=_scores(0.6),
        target=target,
        num_classes=2,
    )
    # 正解観測後にモデル0の領域を学び、次標本では正解を先読みせず選べる。
    second = diagnostics.observe(
        features=torch.tensor([[1.0, 0.0]]),
        predicted_classes=predicted_classes,
        prediction_scores=prediction_scores,
        model_losses=model_losses,
        fallback_scores=_scores(0.6),
        target=target,
        num_classes=2,
    )

    assert first is False
    assert second is True
    assert diagnostics.sample_count == 2
    assert diagnostics.correct_count == 1
    assert diagnostics.fallback_count == 1
    assert diagnostics.selected_count == 1


def test_prototype_routing_discards_shifted_shared_representation():
    diagnostics = PrototypeRoutingDiagnostics()
    diagnostics.fit(
        torch.tensor([[1.0, 0.0]]),
        {0: 0, 1: 1},
        {0: 0.1, 1: 0.9},
    )
    assert diagnostics.prototype_counts[(0, 0)] == 1

    diagnostics.restart_after_aggregation()

    assert diagnostics.prototype_sums == {}
    assert dict(diagnostics.prototype_counts) == {}
