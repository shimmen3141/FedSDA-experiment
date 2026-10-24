"""雑音だけの変更と、通常・固定概念の生成との接続を検証する。"""

import numpy as np

from tools.research_shadow_noise_control import noiseless_sea_policy


def test_noiseless_policy_preserves_features_and_rng_consumption_and_restores():
    from federated_learning_experiments.data.sea.sea_sample_generation import (
        SeaSampleGenerator,
    )

    original = SeaSampleGenerator.generate_sample
    a = SeaSampleGenerator(
        numpy_random_generator=np.random.RandomState(41), concept_count=4
    )
    b = SeaSampleGenerator(
        numpy_random_generator=np.random.RandomState(41), concept_count=4
    )
    c = SeaSampleGenerator(
        numpy_random_generator=np.random.RandomState(41), concept_count=4
    )
    noisy = [a.generate_sample(concept_id=i % 4) for i in range(1000)]
    expected = []
    for i in range(1000):
        x = c.numpy_random_generator.uniform(0.0, 10.0, size=3)
        expected.append(int(x[0] + x[1] <= (9.0, 8.0, 7.0, 9.5)[i % 4]))
        c.numpy_random_generator.rand()
    with noiseless_sea_policy():
        clean = [b.generate_sample(concept_id=i % 4) for i in range(1000)]
    assert SeaSampleGenerator.generate_sample is original
    assert [s.feature_values for s in noisy] == [s.feature_values for s in clean]
    assert [s.class_label for s in clean] == expected
    assert 50 < sum(a.class_label != b.class_label for a, b in zip(noisy, clean)) < 150
    assert (
        a.numpy_random_generator.rand()
        == b.numpy_random_generator.rand()
        == c.numpy_random_generator.rand()
    )


def test_fixed_concept_policy_still_changes_pretraining_concept_without_noise():
    from tools.research_shadow_learner_diagnostic.controls import (
        stationary_concept_policy,
    )
    from federated_learning_experiments.data.sea.sea_sample_generation import (
        SeaSampleGenerator,
    )

    generator = SeaSampleGenerator(
        numpy_random_generator=np.random.RandomState(2), concept_count=4
    )
    with (
        noiseless_sea_policy(),
        stationary_concept_policy(dataset="sea4", concept_id=2),
    ):
        samples = [generator.generate_sample(concept_id=0) for _ in range(100)]
    assert all(
        s.class_label == int(s.feature_values[0] + s.feature_values[1] <= 7.0)
        for s in samples
    )
