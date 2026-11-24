"""割り当てを推定する版のシミュレータ（窓の損失で選び、合うモデルが無ければ新しいモデルを作る）。

simulate.py（割り当てを実験の側で与える版）と同じデータと学習・予測の方式を使い、割り当てだけを推定にする。

割り当ての規則（clientごと、標本1件ごと。予測の後、学習の前）:
- 直近w件の窓（今の標本を含む）で、保有する各モデルの平均二乗損失を計算する。
- まだw件学んでいない若いモデルの値は min(損失, τ) とする（できたばかりのモデルは合っているとみなす。
  そうしないと、学び始めのモデルの損失が大きく、新しいモデルを作り続ける）。
- 値が最小のモデルを選ぶ。最小でもτを超えれば、新しいモデルを作って選ぶ（上限に達していれば最小のモデル）。
- τ = σ²(1 + z√(2/w))。雑音の分散σ²を知っているという単純化（MDLの代わり）。
共有あり（shared）では、同期のとき、モデルIDごとに学習した件数で重み付けて平均し、全clientへ配る。
同期の時点（共有なしでも同じ間隔）で、100件以上学んだモデルの組で、係数の差の2乗 ||w_i − w_j||²
（等方の入力では損失の差の期待値）が (τ − σ²)/4 より小さければ統合する（件数で重み付けた平均。IDの小さい方へ
まとめる）。共有ありでは全clientの見え方、共有なしではclientごとの見え方で行う。oracleでは統合しない。
基準を τ − σ² にすると、遅れの間の誤った学習で2つの概念の中間に寄ったモデルが両方と近く見え、統合が連鎖して
全概念が1つにまとまった（2026-11-24の確認）。

帰属の保留（pending）: 標本tの学習を、P件後（時刻t+P）の判断で行う（P=0なら保留なし）。
窓の中央付近の標本として学習先を決めることになり、遅れの間の誤った学習を減らす（FedHASPの帰属保留FIFOに当たる）。
予測（固い予測）は、その時点の判断（current）のモデルで行う。

mode="oracle" は、割り当てを真の概念にした対照（概念ごとに、最初に出会ったときにモデルを作る）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.research_linear_concept_pool.simulate import _schedule, _segment_types_and_time_since_change


@dataclass(frozen=True)
class EstimatedCell:
    scale: float
    interval: int
    window: int
    clients: int
    sharing: str  # "shared" or "local"
    mode: str = "estimated"  # "estimated" or "oracle"
    concepts: int = 4
    dimension: int = 10
    noise: float = 1.0
    samples_per_client: int = 3000
    synchronization_interval: int = 50
    pending: int = 0
    threshold_z: float = 3.0
    maximum_models: int = 128

    # simulate._schedule が読む名前
    @property
    def delay(self) -> int:
        return 0


MERGE_MINIMUM_SAMPLES = 100
MIXTURE_LEARNING_RATES = (0.1, 0.3, 1.0, 3.0)
MIXTURE_SHARE_RATES = (0.001, 0.01, 0.05)


def threshold(cell: EstimatedCell) -> float:
    return cell.noise**2 * (1.0 + cell.threshold_z * np.sqrt(2.0 / cell.window))


def simulate_estimated(cell: EstimatedCell, seed: int, learning_rate: float) -> dict[str, np.ndarray]:
    """1 runを流し、方式ごとの標本ごとの二乗損失と、割り当ての診断を返す。

    - hard: (M, T)、mix: (F, M, T)、single: (M, T)
    - assigned_concept: (M, T) 標本の学習に割り当てたモデルの、最後の時点の多数派の真の概念（統合を辿る）
    - models_created, merges: スカラー
    """
    rng = np.random.default_rng(seed)
    M, T, K, d = cell.clients, cell.samples_per_client, cell.concepts, cell.dimension
    concepts = _schedule(cell, rng)  # type: ignore[arg-type]
    types, since = _segment_types_and_time_since_change(concepts)
    truth = cell.scale * np.eye(K, d)
    x = rng.standard_normal((M, T, d))
    y = np.einsum("mtd,mtd->mt", x, truth[concepts]) + cell.noise * rng.standard_normal((M, T))

    G = cell.maximum_models
    w = cell.window
    tau = threshold(cell)
    shared = cell.sharing == "shared"
    mix_rates = np.repeat(np.array(MIXTURE_LEARNING_RATES), len(MIXTURE_SHARE_RATES))
    share_rates = np.tile(np.array(MIXTURE_SHARE_RATES), len(MIXTURE_LEARNING_RATES))
    F = len(mix_rates)
    clients = np.arange(M)

    pool = np.zeros((M, G, d))
    exists = np.zeros((M, G), dtype=bool)
    age = np.zeros((M, G))  # そのclientが知る、モデルの学習の件数（共有なら直前の同期までの全体＋自分の分）
    global_count = np.zeros(G)
    trained_since_sync = np.zeros((M, G))
    concept_counts = np.zeros((M, G, K))  # モデルの学習の標本の真の概念の数（多数派の判定用）
    parent = np.tile(np.arange(G), (M, 1))  # 統合の行き先（clientごとの見え方。sharedでは全行が同じ）
    next_global_id = 0
    next_local_id = np.zeros(M, dtype=np.int64)
    current = np.full(M, -1, dtype=np.int64)
    window_x = np.zeros((M, w, d))
    window_y = np.zeros((M, w))
    filled = 0
    log_weight = np.zeros((F, M, G))
    single = np.zeros((M, d))
    models_created = 0
    merges = 0
    pool_uploaded = 0.0
    pool_downloaded = 0.0
    single_communicated = 0.0
    decision_mac = 0.0

    hard_loss = np.zeros((M, T))
    mix_loss = np.zeros((F, M, T))
    single_loss = np.zeros((M, T))
    assigned_id = np.zeros((M, T), dtype=np.int64)
    active_count = np.zeros((M, T))

    def activate(m: int, k: int) -> None:
        count = exists[m].sum() + 1
        log_weight[:, m, k] = np.log(share_rates / count)
        exists[m, k] = True

    def merge_similar_models(rows: np.ndarray) -> int:
        """rowsのclientが共通に見るモデルの集まりで、熟した組の係数の差の2乗が (τ − σ²)/4 より小さければ統合する。

        熟した: その見え方での学習の件数が MERGE_MINIMUM_SAMPLES 以上（学び始めのモデルは係数が0に近く、
        互いに近く見えるため）。件数で重み付けた平均にし、IDの小さい方へまとめる。
        """
        view = rows[0]
        counts = age[view]
        candidates = np.flatnonzero(exists[view] & (counts >= MERGE_MINIMUM_SAMPLES))
        merged = 0
        for i_index, i in enumerate(candidates):
            if not exists[view, i]:
                continue
            for j in candidates[i_index + 1 :]:
                if not exists[view, j]:
                    continue
                if np.sum((pool[view, i] - pool[view, j]) ** 2) >= (tau - cell.noise**2) / 4:
                    continue
                n_i, n_j = age[view, i], age[view, j]
                pool[rows, i] = (n_i * pool[rows, i] + n_j * pool[rows, j]) / (n_i + n_j)
                age[rows, i] = n_i + n_j
                age[rows, j] = 0
                if shared:
                    global_count[i] += global_count[j]
                    global_count[j] = 0
                log_weight[:, rows, i] = np.logaddexp(log_weight[:, rows, i], log_weight[:, rows, j])
                exists[rows, j] = False
                for m in rows:
                    if current[m] == j:
                        current[m] = i
                parent[rows, j] = i
                merged += 1
        return merged

    for t in range(T):
        xt, yt, ct = x[:, t], y[:, t], concepts[:, t]
        predictions = np.einsum("mgd,md->mg", pool, xt)
        losses = (predictions - yt[:, None]) ** 2
        has_current = current >= 0
        hard_prediction = np.where(has_current, predictions[clients, np.maximum(current, 0)], 0.0)
        hard_loss[:, t] = (hard_prediction - yt) ** 2

        mask = exists[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            masked = np.where(mask, log_weight, -np.inf)
            top = masked.max(axis=-1, keepdims=True)
            top = np.where(np.isfinite(top), top, 0.0)
            weights = np.where(mask, np.exp(masked - top), 0.0)
            total = weights.sum(axis=-1, keepdims=True)
            normalized = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
            mixed = np.einsum("fmg,mg->fm", normalized, predictions)
            mix_loss[:, :, t] = (mixed - yt[None]) ** 2
            log_updated = np.where(mask, np.log(normalized) - mix_rates[:, None, None] * losses[None], -np.inf)
            top = log_updated.max(axis=-1, keepdims=True)
            top = np.where(np.isfinite(top), top, 0.0)
            updated = np.where(mask, np.exp(log_updated - top), 0.0)
            updated_total = updated.sum(axis=-1, keepdims=True)
            updated = np.divide(updated, updated_total, out=np.zeros_like(updated), where=updated_total > 0)
            count = np.maximum(exists.sum(axis=1), 1)[None, :, None]
            alpha = share_rates[:, None, None]
            shared_weight = np.where(mask, (1 - alpha) * updated + alpha / count, 0.0)
            log_weight = np.where(mask, np.log(np.maximum(shared_weight, 1e-300)), log_weight)
        active_count[:, t] = exists.sum(axis=1)

        single_prediction = np.einsum("md,md->m", single, xt)
        single_loss[:, t] = (single_prediction - yt) ** 2

        # 窓へ入れる（古い順に並べ、末尾が今の標本）。
        window_x = np.roll(window_x, -1, axis=1)
        window_y = np.roll(window_y, -1, axis=1)
        window_x[:, -1] = xt
        window_y[:, -1] = yt
        filled = min(filled + 1, w)

        # 割り当て。
        if cell.mode == "oracle":
            for m in range(M):
                k = int(ct[m])
                if not exists[m, k]:
                    activate(m, k)
                current[m] = k
        else:
            recent_x = window_x[:, w - filled :]
            recent_y = window_y[:, w - filled :]
            # 保有するモデルの列だけを計算する（上限まで全部を計算すると遅い）。
            columns = np.flatnonzero(exists.any(axis=0))
            window_loss = np.full((M, G), np.inf)
            decision_mac += float(filled * len(columns) * d)
            if len(columns):
                window_predictions = np.einsum("mwd,mgd->mgw", recent_x, pool[:, columns])
                window_loss[:, columns] = ((window_predictions - recent_y[:, None, :]) ** 2).mean(axis=2)
            score = np.where(age < w, np.minimum(window_loss, tau), window_loss)
            score = np.where(exists, score, np.inf)
            best = score.argmin(axis=1)
            best_score = score[clients, best]
            for m in range(M):
                if best_score[m] <= tau:
                    current[m] = best[m]
                    continue
                if shared:
                    if next_global_id < G:
                        k = next_global_id
                        next_global_id += 1
                    else:
                        k = -1
                else:
                    if next_local_id[m] < G:
                        k = int(next_local_id[m])
                        next_local_id[m] += 1
                    else:
                        k = -1
                if k < 0:
                    current[m] = best[m] if np.isfinite(best_score[m]) else 0
                    continue
                pool[m, k] = 0.0
                age[m, k] = 0
                activate(m, k)
                models_created += 1
                current[m] = k
        # 学習（SGDで1回）。保留したP件前の標本を、今の判断のモデルで学ぶ。
        single -= learning_rate * (single_prediction - yt)[:, None] * xt
        trained = t - cell.pending
        if trained >= 0:
            assigned_id[:, trained] = current
            x_trained, y_trained, c_trained = x[:, trained], y[:, trained], concepts[:, trained]
            residual = np.einsum("md,md->m", pool[clients, current], x_trained) - y_trained
            pool[clients, current] -= learning_rate * residual[:, None] * x_trained
            age[clients, current] += 1
            trained_since_sync[clients, current] += 1
            concept_counts[clients, current, c_trained] += 1

        if (t + 1) % cell.synchronization_interval == 0:
            if shared:
                totals = trained_since_sync.sum(axis=0)
                pool_uploaded += float((trained_since_sync > 0).sum() * d) / M
                single_communicated += float(d)
                for k in np.flatnonzero(totals > 0):
                    weight = trained_since_sync[:, k] / totals[k]
                    pool[:, k] = weight @ pool[:, k]
                single[:] = single.mean(axis=0, keepdims=True)
                global_count += totals
                visible = exists.any(axis=0)
                for m, k in zip(*np.nonzero(visible[None] & ~exists), strict=True):
                    activate(int(m), int(k))
                age[:] = global_count[None]
                pool_downloaded += float(exists.any(axis=0).sum() * d)
                views = [clients]
            else:
                views = [np.array([m]) for m in range(M)]
            trained_since_sync[:] = 0
            if cell.mode != "oracle":
                for rows in views:
                    merges += merge_similar_models(rows)

    # 最後まで学ばなかった保留の標本は、最後の判断のモデルに割り当てたとみなす（診断用）。
    if cell.pending > 0:
        assigned_id[:, T - cell.pending :] = current[:, None]

    # 最後の時点の多数派の概念（統合を辿る）。
    def root(m: int, k: int) -> int:
        while parent[m, k] != k:
            k = int(parent[m, k])
        return k

    roots = np.array([[root(m, k) for k in range(G)] for m in range(M)])  # (M, G)
    if shared:
        counts = np.zeros((G, K))
        np.add.at(counts, roots[0], concept_counts.sum(axis=0))
        majority = counts.argmax(axis=1)
        assigned_root = roots[0][assigned_id]
        assigned_concept = majority[assigned_root]
        distinct_models = float(len(np.unique(assigned_root)))
    else:
        assigned_concept = np.zeros((M, T), dtype=np.int64)
        distinct = []
        for m in range(M):
            counts = np.zeros((G, K))
            np.add.at(counts, roots[m], concept_counts[m])
            assigned_root = roots[m][assigned_id[m]]
            assigned_concept[m] = counts.argmax(axis=1)[assigned_root]
            distinct.append(len(np.unique(assigned_root)))
        distinct_models = float(np.mean(distinct))

    return {
        "hard": hard_loss,
        "mix": mix_loss,
        "single": single_loss,
        "types": types,
        "since": since,
        "concepts": concepts,
        "assigned_concept": assigned_concept,
        "active_models": active_count,
        "models_created": np.array(models_created),
        "merges": np.array(merges),
        "pool_uploaded_parameters": np.array(pool_uploaded),
        "pool_downloaded_parameters": np.array(pool_downloaded),
        "single_parameters_each_way": np.array(single_communicated),
        "decision_mac_per_sample": np.array(decision_mac / T),
        "distinct_models": np.array(distinct_models),
        "mix_rates": mix_rates,
        "share_rates": share_rates,
    }
