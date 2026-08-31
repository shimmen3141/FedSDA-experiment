#!/usr/bin/env bash
set -euo pipefail

# 論文の主要構成から一要素ずつ外した、未実施のablationだけを実行する。
# 個別実行: bash tools/run_main_ablation_suite.sh independent
# 一括実行: bash tools/run_main_ablation_suite.sh all

usage() {
    cat <<'EOF'
Usage: bash tools/run_main_ablation_suite.sh [--list] <variant|all> [variant ...]

Variants:
  independent         共有表現を外し、独立モデルにする
  shared-backbone     概念別Residual Adapterを外し、完全共有表現にする
  hard-routing        SoftRoutingを外し、単一モデルを選択する
  global-routing      Meta-switchingを外し、全体損失だけでSoftRoutingする
  switching-routing   モデル追従Fixed-Shareを直接SoftRoutingに使う
  meta-routing        上位switchingを外し、クラス文脈Meta-routerだけを使う
  no-recalibration    集約後FIFO再較正を外す
  immediate-creation  forward検証を外し、警報時に即座に新規モデルを作る
  distance-average    class-functional判定を距離判定へ置き換える
  overall-esr         クラス別損失系列を外し、全体損失e-SRだけを使う
  overall-adwin       クラス別損失系列を外し、全体損失ADWINだけを使う

FDE_RUN_DIRを省略すると、全variantを同じ日時付き結果ルートへ保存します。
FDE_WORKERSの既定値は14です。二つのtmuxで並列に実行する場合は各7以下にしてください。
EOF
}

variants=(
    independent
    shared-backbone
    hard-routing
    global-routing
    switching-routing
    meta-routing
    no-recalibration
    immediate-creation
    distance-average
    overall-esr
    overall-adwin
)

if [[ ${1:-} == --list ]]; then
    printf '%s\n' "${variants[@]}"
    exit 0
fi
if [[ $# -eq 0 || ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    [[ $# -eq 0 ]] && exit 2 || exit 0
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export FDE_WORKERS=${FDE_WORKERS:-14}
export FDE_RUN_DIR=${FDE_RUN_DIR:-"$repo_root/results/results_$(date +%Y%m%d_%H%M%S)_main-ablation-suite"}

base_args=(
    --datasets sea2 sea4 circle2 sine2 mnist2 mnist4
    --concept-schedule random
    --seeds 0 1 2 3 4
    --total-data 5000
    --no-adwin-sweep
    --aggregation-intervals 50 100 200 500
    --fedsda-distance-threshold 0.1
    --clustering-policy on_new_model
    --clustering-consolidation merge
    --no-detection-episodes
    --fifo-size 30
    --no-feddrift
    --no-baselines
    --duplicate-policy error
)
if [[ ${FDE_PLAN_ONLY:-0} == 1 ]]; then
    # エージェントによる内部整合性確認専用。結果ディレクトリは作成しない。
    base_args+=(--print-plan)
fi

final_clustering=(
    --clustering-decision class_functional_confidence
    --cluster-linkage average
)
final_creation=(
    --new-model-creation-policy forward_persistent
    --new-model-forward-validation-samples 10
)
shared_model=(
    --shared-backbone-training joint
    --shared-backbone-gradient-strategy mean
)
residual_model=(
    "${shared_model[@]}"
    --shared-adapter-rank 8
)
final_soft_routing=(
    --shared-backbone-routing-recalibration fifo_replay
    --soft-routing-context meta_switching
    --soft-routing-top-combination leader
    --soft-routing-meta-loss zero_one
    --routing-active-set-policy all
    --no-routing-archive-shadow-diagnostics
)

is_known_variant() {
    local requested=$1
    local candidate
    for candidate in "${variants[@]}"; do
        [[ $requested == "$candidate" ]] && return 0
    done
    return 1
}

is_complete() {
    local manifest="$FDE_RUN_DIR/$1/manifest.json"
    [[ -f $manifest ]] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"completed"' "$manifest"
}

run_variant() {
    local variant=$1
    if is_complete "$variant"; then
        echo "Skip completed variant: $variant ($FDE_RUN_DIR/$variant/manifest.json)"
        return
    fi

    case "$variant" in
        independent)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_clustering[@]}" "${final_creation[@]}" \
                --fedsda-modes FedSDA_NoCached_ClassESR_RestartingSoftRouting \
                --soft-routing-context meta_switching \
                --soft-routing-top-combination leader \
                --soft-routing-meta-loss zero_one \
                --routing-active-set-policy all \
                --no-routing-archive-shadow-diagnostics
            ;;
        shared-backbone)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_clustering[@]}" "${final_creation[@]}" \
                "${shared_model[@]}" "${final_soft_routing[@]}" \
                --fedsda-modes FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting
            ;;
        hard-routing)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_clustering[@]}" "${final_creation[@]}" \
                "${residual_model[@]}" \
                --fedsda-modes FedSDA_NoCached_ResidualAdapter_ClassESR
            ;;
        global-routing)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_clustering[@]}" "${final_creation[@]}" \
                "${residual_model[@]}" \
                --fedsda-modes FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting \
                --shared-backbone-routing-recalibration fifo_replay \
                --soft-routing-context global \
                --routing-active-set-policy all \
                --no-routing-archive-shadow-diagnostics
            ;;
        switching-routing)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_clustering[@]}" "${final_creation[@]}" \
                "${residual_model[@]}" \
                --fedsda-modes FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting \
                --shared-backbone-routing-recalibration fifo_replay \
                --soft-routing-context switching \
                --routing-active-set-policy all \
                --no-routing-archive-shadow-diagnostics
            ;;
        meta-routing)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_clustering[@]}" "${final_creation[@]}" \
                "${residual_model[@]}" \
                --fedsda-modes FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting \
                --shared-backbone-routing-recalibration fifo_replay \
                --soft-routing-context meta_predicted_class \
                --soft-routing-meta-loss zero_one \
                --routing-active-set-policy all \
                --no-routing-archive-shadow-diagnostics
            ;;
        no-recalibration)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_clustering[@]}" "${final_creation[@]}" \
                "${residual_model[@]}" \
                --fedsda-modes FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting \
                --shared-backbone-routing-recalibration none \
                --soft-routing-context meta_switching \
                --soft-routing-top-combination leader \
                --soft-routing-meta-loss zero_one \
                --routing-active-set-policy all \
                --no-routing-archive-shadow-diagnostics
            ;;
        immediate-creation)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_clustering[@]}" "${residual_model[@]}" \
                "${final_soft_routing[@]}" \
                --fedsda-modes FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting \
                --new-model-creation-policy immediate
            ;;
        distance-average)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_creation[@]}" "${residual_model[@]}" \
                "${final_soft_routing[@]}" \
                --fedsda-modes FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting \
                --clustering-decision distance \
                --cluster-linkage average
            ;;
        overall-esr)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_clustering[@]}" "${final_creation[@]}" \
                "${residual_model[@]}" "${final_soft_routing[@]}" \
                --fedsda-modes FedSDA_NoCached_ResidualAdapter_ESR_RestartingSoftRouting
            ;;
        overall-adwin)
            bash "$repo_root/tools/run_server_sweep.sh" "$variant" \
                "${base_args[@]}" "${final_clustering[@]}" "${final_creation[@]}" \
                "${residual_model[@]}" "${final_soft_routing[@]}" \
                --fedsda-modes FedSDA_NoCached_ResidualAdapter_ADWIN_RestartingSoftRouting \
                --fixed-adwin-delta 0.05
            ;;
    esac
}

requested=("$@")
if [[ ${requested[0]} == all ]]; then
    [[ ${#requested[@]} -eq 1 ]] || { echo "all cannot be combined with other variants" >&2; exit 2; }
    requested=("${variants[@]}")
fi

for variant in "${requested[@]}"; do
    is_known_variant "$variant" || { echo "Unknown variant: $variant" >&2; usage >&2; exit 2; }
done
for variant in "${requested[@]}"; do
    run_variant "$variant"
done
