"""同期の版と手元の版の混合（FedHASPの「2つの版の混合」）の利得を、線形とMLPで比べる単純なシミュレータ。

製品のコードもtorchも使わない（numpyだけ。手元のWindowsではtorchのDLLの読み込みが拒否されるため）。

設定（概念は1つで変化しない。FedHASPの「最初の変化の前」に当たる）:
- 入力 x ~ N(0, I_d)、2クラス。ラベルは 1[sin(2x₀) + x₁² − 1 + 0.5x₂ > 0] を確率 flip で反転。
- M client が同じ分布から標本を1件ずつ受け取る。事前学習（全clientで共通の初期モデルを、別の500件で10 epoch）。
- 各時刻: 予測（同期の版、手元の版、2つの混合）→ 標本を保持 → 保持した標本から32件で1回学習（Adam、amsgrad）。
- R件ごとに、手元の版のパラメータを全clientで平均し、同期の版として配る（Adamの状態は手元に残す）。
- 混合: 2つの版の予測確率をclientごとの重みで混ぜる。重みは対数損失のFixed Share（η=1、共有率α）。
- 比べる: 同期の版だけ、手元の版だけ、混合、重み1/2の固定平均。
モデルは線形（ロジスティック回帰）とMLP（隠れ層32・32、ReLU）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VersionCell:
    model: str  # "linear" or "mlp"
    synchronization_interval: int
    learning_rate: float
    clients: int = 10
    dimension: int = 10
    samples_per_client: int = 3000
    label_flip: float = 0.1
    hidden: int = 32
    batch: int = 32
    weight_decay: float = 0.001
    pretraining_samples: int = 500
    pretraining_epochs: int = 10
    share_rate: float = 1.0 / 120.0


def labels(x: np.ndarray, rng: np.random.Generator, flip: float) -> np.ndarray:
    clean = (np.sin(2 * x[..., 0]) + x[..., 1] ** 2 - 1 + 0.5 * x[..., 2] > 0).astype(np.int64)
    flipped = rng.random(clean.shape) < flip
    return np.where(flipped, 1 - clean, clean)


class StackedNetwork:
    """client の次元を先頭に持つパラメータの組。線形は1層、MLPは3層。"""

    def __init__(self, cell: VersionCell, rng: np.random.Generator):
        d, h, c = cell.dimension, cell.hidden, 2
        if cell.model == "linear":
            shapes = [(d, c), (c,)]
        else:
            shapes = [(d, h), (h,), (h, h), (h,), (h, c), (c,)]
        self.params = []
        for shape in shapes:
            if len(shape) == 2:
                bound = 1 / np.sqrt(shape[0])
                self.params.append(rng.uniform(-bound, bound, size=shape))
            else:
                self.params.append(np.zeros(shape))
        self.model = cell.model

    def broadcast(self, clients: int) -> list[np.ndarray]:
        return [np.repeat(p[None], clients, axis=0) for p in self.params]


def forward(params: list[np.ndarray], x: np.ndarray, model: str):
    """params: 各 (M, ...)、x: (M, B, d)。返す: 確率 (M, B, 2) と、逆伝播の中間値。"""
    if model == "linear":
        logits = np.einsum("mbd,mdc->mbc", x, params[0]) + params[1][:, None]
        cache = (x,)
    else:
        a1 = np.einsum("mbd,mdh->mbh", x, params[0]) + params[1][:, None]
        h1 = np.maximum(a1, 0)
        a2 = np.einsum("mbh,mhk->mbk", h1, params[2]) + params[3][:, None]
        h2 = np.maximum(a2, 0)
        logits = np.einsum("mbk,mkc->mbc", h2, params[4]) + params[5][:, None]
        cache = (x, a1, h1, a2, h2)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities, cache


def gradients(params, x, y, model):
    """平均の交差エントロピーの勾配（clientごと）。"""
    probabilities, cache = forward(params, x, model)
    batch = x.shape[1]
    delta = probabilities.copy()
    np.put_along_axis(delta, y[..., None], np.take_along_axis(delta, y[..., None], axis=-1) - 1, axis=-1)
    delta /= batch
    if model == "linear":
        return [np.einsum("mbd,mbc->mdc", x, delta), delta.sum(axis=1)]
    x, a1, h1, a2, h2 = cache
    g5 = np.einsum("mbk,mbc->mkc", h2, delta)
    g6 = delta.sum(axis=1)
    d2 = np.einsum("mbc,mkc->mbk", delta, params[4]) * (a2 > 0)
    g3 = np.einsum("mbh,mbk->mhk", h1, d2)
    g4 = d2.sum(axis=1)
    d1 = np.einsum("mbk,mhk->mbh", d2, params[2]) * (a1 > 0)
    g1 = np.einsum("mbd,mbh->mdh", x, d1)
    g2 = d1.sum(axis=1)
    return [g1, g2, g3, g4, g5, g6]


class AmsgradAdam:
    """torch.optim.Adam(amsgrad=True, weight_decay) と同じ更新（要素ごとなのでclientを重ねてよい）。"""

    def __init__(self, params, learning_rate, weight_decay, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.v_max = [np.zeros_like(p) for p in params]
        self.step_count = 0

    def step(self, params, grads):
        self.step_count += 1
        b1, b2 = self.beta1, self.beta2
        correction1 = 1 - b1**self.step_count
        correction2 = 1 - b2**self.step_count
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            g = g + self.weight_decay * p
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            self.v_max[i] = np.maximum(self.v_max[i], self.v[i])
            denominator = np.sqrt(self.v_max[i]) / np.sqrt(correction2) + self.epsilon
            p -= self.learning_rate / correction1 * self.m[i] / denominator


def bayes_probability_of_class_one(x: np.ndarray, flip: float) -> np.ndarray:
    """ベイズ最適の確率 P(y=1|x)。きれいなラベルが1なら 1−flip、0なら flip。"""
    clean = np.sin(2 * x[..., 0]) + x[..., 1] ** 2 - 1 + 0.5 * x[..., 2] > 0
    return np.where(clean, 1 - flip, flip)


def simulate_versions(
    cell: VersionCell, seed: int, measure_every: int = 0, test_points: int = 1000
) -> dict[str, np.ndarray]:
    """measure_every > 0 なら、その件数ごとに、固定した試験点で各版の確率予測のベイズ最適からのずれを測る。

    試験点は別の乱数（seed + 10**6）で作るので、学習と予測の乱数の消費は measure_every によらず同じ。
    計測（時刻ごと、試験点とclientで平均）:
    - local_excess: E_m E_x[(p_m(x) − p*(x))²]（手元の版のBrierの超過）
    - sync_excess: E_x[(p_sync(x) − p*(x))²]（同期の版。全clientで同じ）
    - local_variance: E_x Var_m(p_m(x))（client間のばらつき）
    - local_mean_bias: E_x[(p̄(x) − p*(x))²]（clientの予測の平均の超過）
    恒等式 local_excess = local_mean_bias + local_variance が成り立つ。
    """
    rng = np.random.default_rng(seed)
    M, T, d = cell.clients, cell.samples_per_client, cell.dimension
    network = StackedNetwork(cell, rng)

    # 事前学習（1つのモデル。M=1として同じ関数を使う）。
    pre_x = rng.standard_normal((cell.pretraining_samples, d))
    pre_y = labels(pre_x, rng, cell.label_flip)
    pre_params = [p[None].copy() for p in network.params]
    pre_optimizer = AmsgradAdam(pre_params, cell.learning_rate, cell.weight_decay)
    for _ in range(cell.pretraining_epochs):
        order = rng.permutation(cell.pretraining_samples)
        for start in range(0, cell.pretraining_samples, cell.batch):
            index = order[start : start + cell.batch]
            pre_optimizer.step(pre_params, gradients(pre_params, pre_x[None, index], pre_y[None, index], cell.model))

    local = [np.repeat(p, M, axis=0) for p in pre_params]
    synchronized = [p.copy() for p in local]
    optimizer = AmsgradAdam(local, cell.learning_rate, cell.weight_decay)
    x = rng.standard_normal((M, T, d))
    y = labels(x, rng, cell.label_flip)
    log_weight = np.zeros((M, 2))  # [同期の版, 手元の版]
    alpha = cell.share_rate
    clients = np.arange(M)

    records = {name: np.zeros((M, T)) for name in ("loss_sync", "loss_local", "loss_mix", "loss_average")}
    records.update(
        {
            name: np.zeros((M, T), dtype=bool)
            for name in ("correct_sync", "correct_local", "correct_mix", "correct_average")
        }
    )
    records["weight_local"] = np.zeros((M, T))
    if measure_every > 0:
        test_rng = np.random.default_rng(seed + 10**6)
        test_x = test_rng.standard_normal((test_points, d))
        bayes = bayes_probability_of_class_one(test_x, cell.label_flip)
        test_batch = np.repeat(test_x[None], M, axis=0)
        measures = {
            name: []
            for name in ("time", "steps_since_sync", "local_excess", "sync_excess", "local_variance", "local_mean_bias")
        }

    for t in range(T):
        xt, yt = x[:, t : t + 1], y[:, t]
        p_sync = forward(synchronized, xt, cell.model)[0][:, 0]
        p_local = forward(local, xt, cell.model)[0][:, 0]
        weights = np.exp(log_weight - log_weight.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        p_mix = weights[:, :1] * p_sync + weights[:, 1:] * p_local
        p_average = 0.5 * (p_sync + p_local)
        for name, p in (("sync", p_sync), ("local", p_local), ("mix", p_mix), ("average", p_average)):
            records[f"loss_{name}"][:, t] = -np.log(np.maximum(p[clients, yt], 1e-12))
            records[f"correct_{name}"][:, t] = p.argmax(axis=1) == yt
        records["weight_local"][:, t] = weights[:, 1]
        # Fixed Share（対数損失、η=1）: 尤度を掛けて正規化し、共有する。
        likelihood = np.stack([p_sync[clients, yt], p_local[clients, yt]], axis=1)
        updated = weights * likelihood
        updated /= updated.sum(axis=1, keepdims=True)
        updated = (1 - alpha) * updated + alpha / 2
        log_weight = np.log(updated)

        # 学習: 保持した標本（0..t）から32件。
        index = rng.integers(0, t + 1, size=(M, cell.batch))
        batch_x = x[clients[:, None], index]
        batch_y = y[clients[:, None], index]
        optimizer.step(local, gradients(local, batch_x, batch_y, cell.model))

        if measure_every > 0 and (t + 1) % measure_every == 0:
            # 学習の後・同期の前の状態を測る（同期の時点では手元の版と同期の版が一致するため）。
            local_probability = forward(local, test_batch, cell.model)[0][..., 1]  # (M, X)
            sync_probability = forward([q[:1] for q in synchronized], test_x[None], cell.model)[0][0, :, 1]
            mean_probability = local_probability.mean(axis=0)
            measures["time"].append(t + 1)
            measures["steps_since_sync"].append(
                (t + 1) % cell.synchronization_interval or cell.synchronization_interval
            )
            measures["local_excess"].append(float(((local_probability - bayes) ** 2).mean()))
            measures["sync_excess"].append(float(((sync_probability - bayes) ** 2).mean()))
            measures["local_variance"].append(float(local_probability.var(axis=0).mean()))
            measures["local_mean_bias"].append(float(((mean_probability - bayes) ** 2).mean()))

        if (t + 1) % cell.synchronization_interval == 0:
            for p in local:
                p[:] = p.mean(axis=0, keepdims=True)
            synchronized = [p.copy() for p in local]

    if measure_every > 0:
        for name, values in measures.items():
            records[f"measure_{name}"] = np.array(values)
    return records
