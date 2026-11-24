"""線形回帰の概念の切り替えで、モデルの集まりの予測の方式を比べる単純なシミュレータ。

製品のコードを使わない（numpyだけ）。理論の見通し（Fixed-Shareの保証、遅れ・誤った割り当ての費用、
概念ごとの学習の費用を全clientで1回だけ払うこと）と1対1で照らすための、最も単純な設定。

設定:
- 入力 x ~ N(0, I_d)。概念kの真の係数は w_k = s * e_k（直交）。y = w_k・x + ε、ε ~ N(0, σ²)。
  概念jのモデルが完全なとき、概念kの標本での損失の差は ||w_j − w_k||² = 2s²。
- 各clientの系列は長さIの区間の並び。最初は概念0、以後は今と違う概念を一様に選ぶ。区間の境はclientごとにずらす。
- 学習の割り当て（全方式で共通）: 真の概念。ただし、変化からdelay件は前の割り当てのまま（検出の遅れ）。
  遅れの後、変化ごとに確率pで、真の概念と違う保有モデル（既に学習されたもの）へ、その区間の終わりまで割り当てる。
- 学習: 割り当てたモデルを標本1件ごとにSGDで1回更新する。単一モデルは全標本で更新する。
- 連合学習（shared）: R件ごとに、モデルIDごとに、その周期に学習した件数で重み付けた平均を全clientへ配る。
  単一モデルはclientの平均。local（単独）は平均しない。
- 予測（学習の前に）:
  - hard: 割り当てたモデル1つ
  - oracle: 真の概念のモデル1つ（同じ学習の集まりから選ぶ。Fixed-Shareの保証の比較相手）
  - mix: 保有モデル全部をFixed-Shareで混ぜる（重みは損失のexp、共有率α）
  - single: 単一モデル
学習率・混合の設定は、複数の値を同時に計算する（学習は予測の方式に依らないので、1回のrunで全ての組を出せる）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEGMENT_TYPE_NAMES = ("first", "recurring", "seen_by_others", "globally_new")
TIME_BIN_EDGES = (0, 5, 10, 20, 50, 100, 200, 10**9)
TIME_BIN_NAMES = ("0-4", "5-9", "10-19", "20-49", "50-99", "100-199", "200-")


@dataclass(frozen=True)
class Cell:
    scale: float
    interval: int
    delay: int
    misassignment_probability: float
    clients: int
    sharing: str  # "shared" or "local"
    concepts: int = 4
    dimension: int = 10
    noise: float = 1.0
    samples_per_client: int = 3000
    synchronization_interval: int = 50


@dataclass(frozen=True)
class Hyper:
    pool_learning_rates: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1)
    single_learning_rates: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1)
    mixture_learning_rates: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0)
    mixture_share_rates: tuple[float, ...] = (0.001, 0.01, 0.05)


def _schedule(cell: Cell, rng: np.random.Generator) -> np.ndarray:
    """clientごとの真の概念の系列 (M, T)。区間の境はclientごとにずらす。"""
    M, T, I, K = cell.clients, cell.samples_per_client, cell.interval, cell.concepts
    concepts = np.zeros((M, T), dtype=np.int64)
    for m in range(M):
        offset = int(rng.integers(0, I))
        current = 0
        t = 0
        first_end = I + offset if M > 1 else I
        concepts[m, :first_end] = current
        t = first_end
        while t < T:
            choices = [k for k in range(K) if k != current]
            current = int(rng.choice(choices))
            concepts[m, t : t + I] = current
            t += I
    return concepts


def _segment_types_and_time_since_change(concepts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """標本ごとの区間の種類（SEGMENT_TYPE_NAMESの添字）と変化からの経過。

    種類は区間の始まりの時刻で決める: そのclientが経験済みならrecurring、
    他のclientが区間の始まりより前に経験していればseen_by_others、どちらも無ければglobally_new。
    最初の区間はfirst。
    """
    M, T = concepts.shape
    K = int(concepts.max()) + 1
    types = np.zeros((M, T), dtype=np.int64)
    since = np.zeros((M, T), dtype=np.int64)
    # first_time[m, k]: clientmが概念kを最初に経験した時刻（無ければ大きい値）。
    first_time = np.full((M, K), np.iinfo(np.int64).max)
    for m in range(M):
        for k in range(K):
            hits = np.flatnonzero(concepts[m] == k)
            if len(hits):
                first_time[m, k] = hits[0]
    for m in range(M):
        starts = np.concatenate([[0], np.flatnonzero(np.diff(concepts[m])) + 1, [T]])
        for start, end in zip(starts[:-1], starts[1:]):
            k = int(concepts[m, start])
            since[m, start:end] = np.arange(end - start)
            if start == 0:
                kind = 0
            elif first_time[m, k] < start:
                kind = 1
            elif np.any(np.delete(first_time[:, k], m) < start):
                # 他のclientが区間の始まりより前に経験した。
                kind = 2
            else:
                kind = 3
            types[m, start:end] = kind
    return types, since


def _training_assignment(concepts: np.ndarray, cell: Cell, rng: np.random.Generator) -> np.ndarray:
    """学習の割り当て (M, T)。遅れと誤った割り当てを与える。

    誤った割り当ての先は、その時点で保有しているモデルに限る（sharedなら直前の同期までに誰かが学習したモデルと
    自分が学習したモデル、localなら自分が学習したモデル）。時刻順に決める必要があるので、学習済みの集合も同時に追う。
    """
    M, T = concepts.shape
    K = cell.concepts
    assignment = np.zeros((M, T), dtype=np.int64)
    trained = np.zeros((M, K), dtype=bool)
    pending_delay = np.zeros(M, dtype=np.int64)
    wrong_target = np.full(M, -1, dtype=np.int64)
    decided = np.ones(M, dtype=bool)
    synchronized = np.zeros(K, dtype=bool)
    for t in range(T):
        if cell.sharing == "shared" and t % cell.synchronization_interval == 0:
            synchronized = trained.any(axis=0)
        for m in range(M):
            c = int(concepts[m, t])
            if t > 0 and c != concepts[m, t - 1]:
                pending_delay[m] = cell.delay
                wrong_target[m] = -1
                decided[m] = False
            if pending_delay[m] > 0:
                assignment[m, t] = assignment[m, t - 1]
                pending_delay[m] -= 1
                continue
            if not decided[m]:
                decided[m] = True
                if rng.random() < cell.misassignment_probability:
                    visible = trained[m] | synchronized if cell.sharing == "shared" else trained[m]
                    others = [k for k in range(K) if k != c and visible[k]]
                    if others:
                        wrong_target[m] = int(rng.choice(others))
            assignment[m, t] = wrong_target[m] if wrong_target[m] >= 0 else c
        trained[np.arange(M), assignment[:, t]] = True
    return assignment


def simulate(cell: Cell, seed: int, hyper: Hyper | None = None) -> dict[str, np.ndarray]:
    """1 runを流し、方式・設定ごとの標本ごとの二乗損失を返す。

    返す配列（形）:
    - hard, oracle: (E, M, T)。Eは学習率の数。
    - mix: (E, F, M, T)。Fは混合の設定（学習率×共有率）の数。
    - single: (S, M, T)。
    - types, since, concepts, assignment: (M, T)。
    - active_models: (M, T) 予測の時点で保有するモデルの数（mixのMACの計算に使う）。
    - pool_uploaded_parameters, pool_downloaded_parameters, single_parameters_each_way: 1 clientあたりの
      通信の浮動小数点数（モデルの集まりの上り・下り、単一モデルの片道）。localでは0。
    """
    hyper = hyper if hyper is not None else Hyper()
    rng = np.random.default_rng(seed)
    M, T, K, d = cell.clients, cell.samples_per_client, cell.concepts, cell.dimension
    concepts = _schedule(cell, rng)
    types, since = _segment_types_and_time_since_change(concepts)
    assignment = _training_assignment(concepts, cell, rng)
    truth = cell.scale * np.eye(K, d)
    x = rng.standard_normal((M, T, d))
    y = np.einsum("mtd,mtd->mt", x, truth[concepts]) + cell.noise * rng.standard_normal((M, T))

    pool_rates = np.array(hyper.pool_learning_rates)
    single_rates = np.array(hyper.single_learning_rates)
    mix_rates = np.repeat(np.array(hyper.mixture_learning_rates), len(hyper.mixture_share_rates))
    share_rates = np.tile(np.array(hyper.mixture_share_rates), len(hyper.mixture_learning_rates))
    E, S, F = len(pool_rates), len(single_rates), len(mix_rates)

    pool = np.zeros((E, M, K, d))
    single = np.zeros((S, M, d))
    active = np.zeros((M, K), dtype=bool)
    trained_since_sync = np.zeros((M, K))
    log_weight = np.zeros((E, F, M, K))
    clients = np.arange(M)

    hard_loss = np.zeros((E, M, T))
    oracle_loss = np.zeros((E, M, T))
    mix_loss = np.zeros((E, F, M, T))
    single_loss = np.zeros((S, M, T))
    active_count = np.zeros((M, T), dtype=np.int64)
    pool_uploaded = 0.0
    pool_downloaded = 0.0
    single_communicated = 0.0

    for t in range(T):
        xt, yt, a, c = x[:, t], y[:, t], assignment[:, t], concepts[:, t]
        predictions = np.einsum("emkd,md->emk", pool, xt)  # (E, M, K)
        losses = (predictions - yt[None, :, None]) ** 2
        hard_loss[:, :, t] = losses[:, clients, a]
        oracle_loss[:, :, t] = losses[:, clients, c]

        # Fixed-Share: 保有するモデルだけで混ぜる。保有が無いclientは0を予測する。
        mask = active[None, None]  # (1, 1, M, K)
        masked = np.where(mask, log_weight, -np.inf)
        top = masked.max(axis=-1, keepdims=True)
        top = np.where(np.isfinite(top), top, 0.0)
        weights = np.where(mask, np.exp(masked - top), 0.0)
        total = weights.sum(axis=-1, keepdims=True)
        normalized = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
        mixed = np.einsum("efmk,emk->efm", normalized, predictions)
        mix_loss[:, :, :, t] = (mixed - yt[None, None, :]) ** 2
        active_count[:, t] = active.sum(axis=1)

        # 重みの更新（損失のexp）と共有。
        with np.errstate(divide="ignore"):
            log_updated = np.where(mask, np.log(normalized) - mix_rates[None, :, None, None] * losses[:, None], -np.inf)
        top = log_updated.max(axis=-1, keepdims=True)
        top = np.where(np.isfinite(top), top, 0.0)
        updated = np.where(mask, np.exp(log_updated - top), 0.0)
        updated_total = updated.sum(axis=-1, keepdims=True)
        updated = np.divide(updated, updated_total, out=np.zeros_like(updated), where=updated_total > 0)
        count = np.maximum(active.sum(axis=1), 1)[None, None, :, None]
        alpha = share_rates[None, :, None, None]
        shared_weight = np.where(mask, (1 - alpha) * updated + alpha / count, 0.0)
        log_weight = np.where(mask, np.log(np.maximum(shared_weight, 1e-300)), log_weight)

        single_prediction = np.einsum("smd,md->sm", single, xt)
        single_loss[:, :, t] = (single_prediction - yt[None]) ** 2

        # 学習（SGDで1回）。
        residual = predictions[:, clients, a] - yt[None]  # (E, M)
        pool[:, clients, a] -= pool_rates[:, None, None] * residual[:, :, None] * xt[None]
        single -= single_rates[:, None, None] * (single_prediction - yt[None])[:, :, None] * xt[None]
        newly_active = ~active[clients, a]
        if newly_active.any():
            # 新しく保有したモデルの重みは、共有で配る量（α/保有数）から始める。
            m_new = clients[newly_active]
            k_new = a[newly_active]
            count_new = active[m_new].sum(axis=1) + 1
            log_weight[:, :, m_new, k_new] = np.log(share_rates[None, :, None] / count_new[None, None, :])
        active[clients, a] = True
        trained_since_sync[clients, a] += 1

        if cell.sharing == "shared" and (t + 1) % cell.synchronization_interval == 0:
            totals = trained_since_sync.sum(axis=0)  # (K,)
            pool_uploaded += float((trained_since_sync > 0).sum() * d) / M
            single_communicated += float(d)
            for k in np.flatnonzero(totals > 0):
                weight = trained_since_sync[:, k] / totals[k]
                pool[:, :, k] = np.einsum("m,emd->ed", weight, pool[:, :, k])[:, None]
            single[:] = single.mean(axis=1, keepdims=True)
            newly_visible = active.any(axis=0)[None] & ~active
            for m, k in zip(*np.nonzero(newly_visible)):
                count_new = active[m].sum() + 1
                log_weight[:, :, m, k] = np.log(share_rates[None, :] / count_new)
                active[m, k] = True
            pool_downloaded += float(active.any(axis=0).sum() * d)
            trained_since_sync[:] = 0

    return {
        "hard": hard_loss,
        "oracle": oracle_loss,
        "mix": mix_loss,
        "single": single_loss,
        "types": types,
        "since": since,
        "concepts": concepts,
        "assignment": assignment,
        "active_models": active_count,
        "pool_uploaded_parameters": np.array(pool_uploaded),
        "pool_downloaded_parameters": np.array(pool_downloaded),
        "single_parameters_each_way": np.array(single_communicated),
        "pool_rates": pool_rates,
        "single_rates": single_rates,
        "mix_rates": mix_rates,
        "share_rates": share_rates,
    }
