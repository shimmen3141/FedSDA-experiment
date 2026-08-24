"""実験オプションの依存関係・適用可能性・実装状態を管理する。"""

from dataclasses import dataclass


OPTION_SCHEMA_VERSION = 4

FED_SDA = "FedSDA"
FED_DRIFT = "FedDrift"
WITHOUT_SERVER = "FedSDA_without_server"
OBLIVIOUS = "Oblivious"


@dataclass(frozen=True)
class CapabilitySpec:
    id: str
    title: str
    description: str


@dataclass(frozen=True)
class MethodSpec:
    id: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class ActivationRule:
    """別オプションが指定値のときだけ対象オプションを有効にする。"""

    option_id: str
    values: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class OptionSpec:
    id: str
    title: str
    category: str
    choices: tuple[str, ...]
    description: str
    implemented_for: tuple[str, ...]
    theoretically_applicable_to: tuple[str, ...]
    requires_capabilities: tuple[str, ...] = ()
    active_when: tuple[ActivationRule, ...] = ()
    cli_name: str | None = None
    configuration_surface: str = "cli"


@dataclass(frozen=True)
class ChoiceConstraint:
    """特定の選択肢にだけ課される実装上の制約。"""

    option_id: str
    values: tuple[str, ...]
    implemented_modes: tuple[str, ...]
    requires_selections: tuple[ActivationRule, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class SweepDependency:
    """掃引リストが空のとき無効になる固定側CLIオプション。"""

    controller_cli: str
    dependent_cli: str
    description: str


@dataclass(frozen=True)
class CollectionControl:
    """mode集合または掃引値集合の既定・無効化方法。"""

    id: str
    values_cli: str
    disable_cli: str
    default_behavior: str
    legacy_empty_supported: bool = False


CAPABILITIES = (
    CapabilitySpec("sample_wise_detection", "サンプル単位検出", "各到着サンプルで検出器を更新する"),
    CapabilitySpec("batch_wise_detection", "バッチ単位検出", "検出バッチ単位でモデル差を判定する"),
    CapabilitySpec("multiple_models", "複数モデル保持", "概念ごとの複数モデルを保持できる"),
    CapabilitySpec("provisional_model", "仮モデル", "採用前の候補モデルを保持できる"),
    CapabilitySpec("future_sample_validation", "将来サンプル検証", "警報後の到着データで候補を評価できる"),
    CapabilitySpec("server_clustering", "サーバクラスタリング", "サーバでモデル統合を実行できる"),
    CapabilitySpec("soft_routing", "SoftRouting", "保持モデルの予測を重み付き混合できる"),
    CapabilitySpec("shared_representation", "共有表現", "複数概念モデルで特徴抽出層を共有できる"),
)

METHODS = (
    MethodSpec(
        FED_SDA,
        (
            "sample_wise_detection", "multiple_models", "provisional_model",
            "future_sample_validation", "server_clustering", "soft_routing",
            "shared_representation",
        ),
    ),
    MethodSpec(
        FED_DRIFT,
        ("batch_wise_detection", "multiple_models", "server_clustering"),
    ),
    MethodSpec(WITHOUT_SERVER, ("sample_wise_detection", "multiple_models")),
    MethodSpec(OBLIVIOUS, ()),
)

_FORWARD_POLICIES = (
    "forward_validated", "forward_requalified",
    "forward_requalified_current_first", "forward_persistent",
    "shadow_tournament",
)

OPTIONS = (
    OptionSpec(
        "method", "手法", "method",
        (FED_SDA, FED_DRIFT, WITHOUT_SERVER, OBLIVIOUS),
        "実験プロトコルを選択する最上位オプション",
        (FED_SDA, FED_DRIFT, WITHOUT_SERVER, OBLIVIOUS),
        (FED_SDA, FED_DRIFT, WITHOUT_SERVER, OBLIVIOUS),
        cli_name="mode", configuration_surface="mode",
    ),
    OptionSpec(
        "server_flow", "FedSDAサーバフロー", "protocol",
        ("NoCached", "Cached"),
        "FedAvg前後のモデルキャッシュ利用と通信順序",
        (FED_SDA,), (FED_SDA,),
        requires_capabilities=("server_clustering",),
        configuration_surface="mode",
    ),
    OptionSpec(
        "detector", "ドリフト検出器", "detection",
        ("ADWIN", "ClassADWIN", "ESR", "ClassESR", "HDDMA", "ClassHDDMA", "HDDMW"),
        "FedSDAクライアントが損失系列へ適用する検出器",
        (FED_SDA, WITHOUT_SERVER), (FED_SDA, WITHOUT_SERVER),
        requires_capabilities=("sample_wise_detection",),
        configuration_surface="mode",
    ),
    OptionSpec(
        "routing", "予測ルーティング", "prediction",
        ("hard", "restarting_soft", "protected_soft"),
        "保持モデルから予測を選択または混合する方式",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("multiple_models",),
        configuration_surface="mode",
    ),
    OptionSpec(
        "soft_routing_context", "SoftRouting文脈", "prediction",
        (
            "global", "predicted_class", "meta_predicted_class",
            "meta_switching",
        ),
        "大域・予測クラス別・meta混合、またはmetaとswitching-expertの上位選択を選ぶ",
        (FED_SDA,), (FED_SDA,),
        requires_capabilities=("soft_routing",),
        active_when=(ActivationRule(
            "routing", ("restarting_soft",),
            "Restarting SoftRoutingのとき",
        ),),
        cli_name="soft-routing-context",
    ),
    OptionSpec(
        "soft_routing_top_combination", "Meta-switching上位統合", "prediction",
        ("leader", "mixture"),
        "上位Fixed-Shareの最大重み候補を使うか、候補予測を重み付き混合する",
        (FED_SDA,), (FED_SDA,),
        requires_capabilities=("soft_routing",),
        active_when=(ActivationRule(
            "soft_routing_context", ("meta_switching",),
            "Meta-switchingを実予測へ使うとき",
        ),),
        cli_name="soft-routing-top-combination",
    ),
    OptionSpec(
        "soft_routing_meta_loss", "Meta-router更新損失", "prediction",
        ("bounded_score", "zero_one"),
        "Meta候補を確率出力の有界損失または最終予測の0/1損失で比較する",
        (FED_SDA,), (FED_SDA,),
        requires_capabilities=("soft_routing",),
        active_when=(ActivationRule(
            "soft_routing_context",
            ("predicted_class", "meta_predicted_class", "meta_switching"),
            "文脈別Meta-routerを計算するとき",
        ),),
        cli_name="soft-routing-meta-loss",
    ),
    OptionSpec(
        "model_architecture", "モデル構造", "model",
        ("independent", "shared_backbone", "residual_adapter"),
        "概念モデルを独立保持するか、特徴抽出部を共有して概念別ヘッドを持つか",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("multiple_models",),
        configuration_surface="mode",
    ),
    OptionSpec(
        "shared_backbone_training", "共有表現学習", "model",
        ("sequential", "joint", "frozen"),
        "通常ローカル更新で共有部を逐次更新・共同更新・固定のどれにするか",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("shared_representation",),
        active_when=(ActivationRule(
            "model_architecture",
            ("shared_backbone", "residual_adapter"),
            "共有表現構造のとき",
        ),),
        cli_name="shared-backbone-training",
    ),
    OptionSpec(
        "shared_backbone_gradient_strategy", "共有勾配統合", "model",
        ("mean", "pcgrad"),
        "共同学習時の概念別バックボーン勾配を平均または競合射影で統合する方式",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("shared_representation",),
        active_when=(ActivationRule(
            "shared_backbone_training", ("joint",),
            "共有表現をjoint学習するとき",
        ),),
        cli_name="shared-backbone-gradient-strategy",
    ),
    OptionSpec(
        "shared_backbone_routing_recalibration",
        "共有表現更新後のルーティング再較正",
        "prediction",
        (
            "none", "aggregation_restart", "fifo_replay",
            "leader_change_replay", "persistent_leader_change_replay",
        ),
        "サーバ集約で共有表現が変化した後にSoftRoutingの累積証拠を扱う方式",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("shared_representation", "soft_routing"),
        active_when=(ActivationRule(
            "model_architecture",
            ("shared_backbone", "residual_adapter"),
            "共有表現構造のとき",
        ), ActivationRule(
            "routing",
            ("restarting_soft",),
            "Restarting SoftRoutingのとき",
        )),
        cli_name="shared-backbone-routing-recalibration",
    ),
    OptionSpec(
        "shared_adapter_rank", "概念別残差adapter rank R_adapter", "model",
        ("positive integer",),
        "低ランク残差adapterの最大rank。特徴次元を上限とする",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("shared_representation",),
        active_when=(ActivationRule(
            "model_architecture", ("residual_adapter",),
            "低ランク残差adapter構造のとき",
        ),),
        cli_name="shared-adapter-rank",
    ),
    OptionSpec(
        "new_model_creation_policy", "新規モデル作成方針", "adaptation",
        (
            "immediate", "validated", "forward_validated",
            "forward_requalified", "forward_requalified_current_first",
            "forward_persistent", "shadow_tournament",
        ),
        "警報後に新規モデルを即時作成するか、検証してから採用するか",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("multiple_models",),
        cli_name="new-model-creation-policy",
    ),
    OptionSpec(
        "new_model_training", "新規モデル初期学習", "adaptation",
        ("none", "fixed", "early_stopping"),
        "新規モデル候補の初期学習方法",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("multiple_models",),
        cli_name="new-model-training",
    ),
    OptionSpec(
        "new_model_epochs", "新規モデル学習上限 E_init", "adaptation_parameter",
        ("non-negative integer",),
        "fixedのエポック数またはearly_stoppingの最大エポック数",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        active_when=(ActivationRule(
            "new_model_training", ("fixed", "early_stopping"), "初期学習を行うとき",
        ),),
        cli_name="new-model-epochs",
    ),
    OptionSpec(
        "new_model_initialization", "新規モデル初期値", "adaptation",
        ("current", "best_candidate", "average"),
        "新規モデル候補を初期化する既存パラメータの選択方法",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("multiple_models",),
        cli_name="new-model-initialization",
    ),
    OptionSpec(
        "new_model_validation_fraction", "履歴内検証割合", "adaptation_parameter",
        ("0 < fraction < 1",),
        "validated方式でFIFO末尾から検証用に確保する割合",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        active_when=(ActivationRule(
            "new_model_creation_policy", ("validated",), "validatedのとき",
        ),),
        cli_name="new-model-validation-fraction",
    ),
    OptionSpec(
        "new_model_forward_validation_samples", "前向き検証数 N_forward",
        "adaptation_parameter", ("integer >= 2",),
        "forward系方式で警報後に収集する将来サンプル数",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("provisional_model", "future_sample_validation"),
        active_when=(ActivationRule(
            "new_model_creation_policy", _FORWARD_POLICIES, "forward系のとき",
        ),),
        cli_name="new-model-forward-validation-samples",
    ),
    OptionSpec(
        "fifo_size", "FIFO長 N_FIFO", "detection_parameter",
        ("integer >= 1",),
        "FedSDAの検出・ドリフト解決に保持する直近データ数",
        (FED_SDA, WITHOUT_SERVER), (FED_SDA, WITHOUT_SERVER),
        requires_capabilities=("sample_wise_detection",), cli_name="fifo-size",
    ),
    OptionSpec(
        "detection_episodes", "検出エピソード", "detection",
        ("off", "on"),
        "近接した警報をN_FIFO幅の一つの適応エピソードへ統合する",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("sample_wise_detection",), cli_name="detection-episodes",
    ),
    OptionSpec(
        "adwin_delta", "ADWINのδ_ADWIN", "detection_parameter",
        ("0 < delta < 1",), "ADWIN系検出器の信頼度パラメータ",
        (FED_SDA, WITHOUT_SERVER), (FED_SDA, WITHOUT_SERVER),
        active_when=(ActivationRule(
            "detector", ("ADWIN", "ClassADWIN"), "ADWIN系のとき",
        ),),
        cli_name="adwin-deltas",
    ),
    OptionSpec(
        "e_detector_alpha", "e-SRのα_e", "detection_parameter",
        ("0 < alpha < 1",), "ESR系検出器の誤警報制御値",
        (FED_SDA,), (FED_SDA,),
        active_when=(ActivationRule(
            "detector", ("ESR", "ClassESR"), "ESR系のとき",
        ),),
        configuration_surface="config",
    ),
    OptionSpec(
        "hddm_drift_confidence", "HDDMのconfidence", "detection_parameter",
        ("0 < confidence < 1",), "HDDM系検出器のドリフト判定信頼度",
        (FED_SDA,), (FED_SDA,),
        active_when=(ActivationRule(
            "detector", ("HDDMA", "ClassHDDMA", "HDDMW"), "HDDM系のとき",
        ),),
        configuration_surface="config",
    ),
    OptionSpec(
        "clustering_policy", "クラスタリング頻度", "clustering",
        ("disabled", "on_new_model", "every_round"),
        "FedSDAサーバがモデル統合判定を実行するタイミング",
        (FED_SDA,), (FED_SDA,),
        requires_capabilities=("server_clustering",), cli_name="clustering-policy",
    ),
    OptionSpec(
        "clustering_consolidation", "クラスタリング後処理", "clustering",
        ("merge", "parameter_share", "noninferiority_merge"),
        "クラスタ決定後にIDを統合するか、パラメータ共有または非劣性検証を行うか",
        (FED_SDA,), (FED_SDA,),
        requires_capabilities=("server_clustering",),
        active_when=(ActivationRule(
            "clustering_policy", ("on_new_model", "every_round"),
            "クラスタリングを実行するとき",
        ),),
        cli_name="clustering-consolidation",
    ),
    OptionSpec(
        "merge_noninferiority_margin", "統合モデルの非劣性幅", "clustering_parameter",
        ("non-negative number",),
        "仮統合モデルの対応あり損失差について許容する片側上限",
        (FED_SDA,), (FED_SDA,),
        requires_capabilities=("server_clustering",),
        active_when=(ActivationRule(
            "clustering_consolidation", ("noninferiority_merge",),
            "非劣性制約付きマージのとき",
        ),),
        cli_name="merge-noninferiority-margin",
    ),
    OptionSpec(
        "clustering_decision", "クラスタリング判定", "clustering",
        ("distance", "confidence", "confidence_margin", "oracle_concept"),
        "モデル対を統合する判定規則",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        requires_capabilities=("server_clustering",), cli_name="clustering-decision",
    ),
    OptionSpec(
        "clustering_confidence", "クラスタリング信頼水準", "clustering_parameter",
        ("0 < confidence < 1",),
        "confidence系統合判定と非劣性制約付きマージの信頼水準",
        (FED_SDA,), (FED_SDA, FED_DRIFT),
        configuration_surface="config",
    ),
    OptionSpec(
        "cluster_linkage", "階層クラスタリングlinkage", "clustering",
        ("complete", "connected"), "モデル対判定からクラスタを構成する方法",
        (FED_SDA,), (FED_SDA,),
        requires_capabilities=("server_clustering",), cli_name="cluster-linkage",
    ),
    OptionSpec(
        "fedsda_distance_threshold", "FedSDA距離閾値 γ", "clustering_parameter",
        ("non-negative number",), "モデル適合・再利用および距離ベース統合の閾値",
        (FED_SDA,), (FED_SDA,), cli_name="fedsda-distance-threshold",
    ),
    OptionSpec(
        "feddrift_distance_threshold", "FedDrift距離閾値 δ_FedDrift",
        "clustering_parameter", ("non-negative number",),
        "FedDriftのドリフト判定とモデル統合で共有する閾値",
        (FED_DRIFT,), (FED_DRIFT,), cli_name="feddrift-distance-thresholds",
    ),
    OptionSpec(
        "aggregation_interval", "集約間隔 A", "federation_parameter",
        ("positive integer",), "FedSDAとObliviousの通信ラウンド間隔",
        (FED_SDA, WITHOUT_SERVER, OBLIVIOUS),
        (FED_SDA, WITHOUT_SERVER, OBLIVIOUS),
        cli_name="aggregation-intervals",
    ),
    OptionSpec(
        "feddrift_detection_batch_size", "FedDrift検出バッチ B_detect",
        "federation_parameter", ("positive integer",),
        "FedDriftの処理・検出・通信を兼ねるバッチサイズ",
        (FED_DRIFT,), (FED_DRIFT,),
        requires_capabilities=("batch_wise_detection",),
        cli_name="feddrift-detection-batch-sizes",
    ),
    OptionSpec(
        "feddrift_isolation_timesteps", "FedDrift隔離時刻数 W", "adaptation_parameter",
        ("non-negative integer",), "新規FedDriftモデルをマージ対象から外す時刻数",
        (FED_DRIFT,), (FED_DRIFT,), cli_name="feddrift-isolation",
    ),
    OptionSpec(
        "experiment_manifest", "実験manifest", "execution",
        ("off", "on"), "実験計画・実装由来・完了状態を出力先へ保存する",
        (FED_SDA, FED_DRIFT, WITHOUT_SERVER, OBLIVIOUS),
        (FED_SDA, FED_DRIFT, WITHOUT_SERVER, OBLIVIOUS),
        cli_name="manifest",
    ),
    OptionSpec(
        "duplicate_policy", "既存実験重複ポリシー", "execution",
        ("ignore", "warn", "error"),
        "一部でも同一設定・同一コード・同一goldenの完了runがある場合の開始前処理",
        (FED_SDA, FED_DRIFT, WITHOUT_SERVER, OBLIVIOUS),
        (FED_SDA, FED_DRIFT, WITHOUT_SERVER, OBLIVIOUS),
        cli_name="duplicate-policy",
    ),
)

CHOICE_CONSTRAINTS = (
    ChoiceConstraint(
        "clustering_consolidation",
        ("parameter_share", "noninferiority_merge"),
        (
            "FedSDA_NoCached_ADWIN",
            "FedSDA_NoCached_ClassADWIN",
            "FedSDA_NoCached_ESR",
            "FedSDA_NoCached_ClassESR",
            "FedSDA_NoCached_ClassESR_RestartingSoftRouting",
            "FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting",
            "FedSDA_NoCached_ResidualAdapter_ClassADWIN_RestartingSoftRouting",
            "FedSDA_NoCached_ResidualAdapter_ClassESR",
            "FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
            "FedSDA_NoCached_ClassESR_ProtectedSoftRouting",
            "FedSDA_NoCached_HDDMA",
            "FedSDA_NoCached_ClassHDDMA",
            "FedSDA_NoCached_HDDMW",
        ),
        requires_selections=(
            ActivationRule("server_flow", ("NoCached",), "NoCachedが必要"),
        ),
        note="追加のクラスタリング後処理は現在NoCachedフローでのみ実装",
    ),
    ChoiceConstraint(
        "detector", ("ADWIN",),
        ("FedSDA_NoCached_ADWIN", "FedSDA_Cached_ADWIN", "FedSDA_without_server"),
        note="without_serverで選べる検出器は現在ADWINのみ",
    ),
    ChoiceConstraint(
        "detector", ("ClassADWIN",),
        (
            "FedSDA_NoCached_ClassADWIN",
            "FedSDA_Cached_ClassADWIN",
            "FedSDA_NoCached_ResidualAdapter_ClassADWIN_RestartingSoftRouting",
        ),
    ),
    ChoiceConstraint(
        "detector", ("ESR",),
        ("FedSDA_NoCached_ESR", "FedSDA_Cached_ESR"),
    ),
    ChoiceConstraint(
        "detector", ("ClassESR",),
        (
            "FedSDA_NoCached_ClassESR", "FedSDA_Cached_ClassESR",
            "FedSDA_NoCached_ClassESR_RestartingSoftRouting",
            "FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting",
            "FedSDA_NoCached_ResidualAdapter_ClassESR",
            "FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
            "FedSDA_NoCached_ClassESR_ProtectedSoftRouting",
        ),
    ),
    ChoiceConstraint(
        "detector", ("HDDMA",),
        ("FedSDA_NoCached_HDDMA", "FedSDA_Cached_HDDMA"),
    ),
    ChoiceConstraint(
        "detector", ("ClassHDDMA",),
        ("FedSDA_NoCached_ClassHDDMA", "FedSDA_Cached_ClassHDDMA"),
    ),
    ChoiceConstraint(
        "detector", ("HDDMW",),
        ("FedSDA_NoCached_HDDMW", "FedSDA_Cached_HDDMW"),
    ),
    ChoiceConstraint(
        "routing", ("restarting_soft",),
        (
            "FedSDA_NoCached_ClassESR_RestartingSoftRouting",
            "FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting",
            "FedSDA_NoCached_ResidualAdapter_ClassADWIN_RestartingSoftRouting",
            "FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
        ),
        requires_selections=(
            ActivationRule("server_flow", ("NoCached",), "NoCachedが必要"),
            ActivationRule(
                "detector", ("ClassADWIN", "ClassESR"),
                "ClassADWINまたはClassESRが必要",
            ),
        ),
        note="ClassADWINとClassESRの専用modeで実装",
    ),
    ChoiceConstraint(
        "model_architecture", ("shared_backbone",),
        ("FedSDA_NoCached_SharedBackbone_ClassESR_RestartingSoftRouting",),
        requires_selections=(
            ActivationRule("server_flow", ("NoCached",), "NoCachedが必要"),
            ActivationRule("routing", ("restarting_soft",), "Restarting SoftRoutingが必要"),
        ),
        note="共有バックボーンは専用modeで実装",
    ),
    ChoiceConstraint(
        "model_architecture", ("residual_adapter",),
        (
            "FedSDA_NoCached_ResidualAdapter_ClassADWIN_RestartingSoftRouting",
            "FedSDA_NoCached_ResidualAdapter_ClassESR",
            "FedSDA_NoCached_ResidualAdapter_ClassESR_RestartingSoftRouting",
        ),
        requires_selections=(
            ActivationRule("server_flow", ("NoCached",), "NoCachedが必要"),
        ),
        note="低ランク残差adapterはhard routingとRestarting SoftRoutingで実装",
    ),
    ChoiceConstraint(
        "routing", ("protected_soft",),
        ("FedSDA_NoCached_ClassESR_ProtectedSoftRouting",),
        requires_selections=(
            ActivationRule("server_flow", ("NoCached",), "NoCachedが必要"),
            ActivationRule("detector", ("ClassESR",), "ClassESRが必要"),
        ),
        note="現在は専用modeでのみ実装",
    ),
)

SWEEP_DEPENDENCIES = (
    SweepDependency(
        "adwin-deltas", "fixed-aggregation-interval",
        "δ_ADWIN掃引中に固定する集約間隔A",
    ),
    SweepDependency(
        "aggregation-intervals", "fixed-adwin-delta",
        "A掃引中に固定するδ_ADWIN",
    ),
    SweepDependency(
        "feddrift-detection-batch-sizes", "fixed-feddrift-distance-threshold",
        "B_detect掃引中に固定するδ_FedDrift",
    ),
    SweepDependency(
        "feddrift-distance-thresholds", "fixed-feddrift-detection-batch-size",
        "δ_FedDrift掃引中に固定するB_detect",
    ),
)

COLLECTION_CONTROLS = (
    CollectionControl("fedsda_modes", "fedsda-modes", "no-fedsda", "全FedSDA mode"),
    CollectionControl("feddrift_modes", "feddrift-modes", "no-feddrift", "FedDrift"),
    CollectionControl("baseline_modes", "baseline-modes", "no-baselines", "全baseline"),
    CollectionControl(
        "adwin_delta_sweep", "adwin-deltas", "no-adwin-sweep", "既定δ_ADWIN集合",
    ),
    CollectionControl(
        "aggregation_sweep", "aggregation-intervals", "no-aggregation-sweep", "既定A集合",
    ),
    CollectionControl(
        "feddrift_batch_sweep", "feddrift-detection-batch-sizes",
        "no-feddrift-batch-sweep", "既定B_detect集合",
    ),
    CollectionControl(
        "feddrift_distance_sweep", "feddrift-distance-thresholds",
        "no-feddrift-distance-sweep", "既定δ_FedDrift集合",
    ),
)

CAPABILITIES_BY_ID = {item.id: item for item in CAPABILITIES}
METHODS_BY_ID = {item.id: item for item in METHODS}
OPTIONS_BY_ID = {item.id: item for item in OPTIONS}
CHOICE_CONSTRAINTS_BY_OPTION = {
    option_id: tuple(item for item in CHOICE_CONSTRAINTS if item.option_id == option_id)
    for option_id in {item.option_id for item in CHOICE_CONSTRAINTS}
}


def option(option_id):
    try:
        return OPTIONS_BY_ID[option_id]
    except KeyError as exc:
        raise KeyError(f"Unknown option id: {option_id}") from exc


def implementation_status(option_id, method_id):
    """implemented / theoretical / unavailable のいずれかを返す。"""
    spec = option(option_id)
    if method_id in spec.implemented_for:
        return "implemented"
    if method_id in spec.theoretically_applicable_to:
        return "theoretical"
    return "unavailable"


def inactive_reasons(option_id, selections):
    """現在の選択で無効なら、その依存条件を人間向け文字列で返す。"""
    reasons = []
    for rule in option(option_id).active_when:
        if selections.get(rule.option_id) not in rule.values:
            reasons.append(rule.label)
    return tuple(reasons)


def validate_selection(method_id, selections):
    """実装済みの組み合わせかを検証し、問題点を文字列で返す。"""
    if method_id not in METHODS_BY_ID:
        return (f"unknown method: {method_id}",)
    method_capabilities = set(METHODS_BY_ID[method_id].capabilities)
    issues = []
    for option_id, selected_value in selections.items():
        spec = option(option_id)
        status = implementation_status(option_id, method_id)
        if status != "implemented":
            issues.append(f"{option_id} is {status} for {method_id}")
            continue
        missing = set(spec.requires_capabilities) - method_capabilities
        if missing:
            issues.append(f"{option_id} requires: {', '.join(sorted(missing))}")
        issues.extend(f"{option_id} is active only when {reason}"
                      for reason in inactive_reasons(option_id, selections))
        for constraint in CHOICE_CONSTRAINTS_BY_OPTION.get(option_id, ()):
            if selected_value not in constraint.values:
                continue
            for rule in constraint.requires_selections:
                if selections.get(rule.option_id) not in rule.values:
                    issues.append(f"{option_id}={selected_value}: {rule.label}")
    return tuple(issues)


def method_for_mode(mode):
    """具体的なmode名をOptionSpec上の手法へ正規化する。"""
    if mode == FED_DRIFT:
        return FED_DRIFT
    if mode == WITHOUT_SERVER:
        return WITHOUT_SERVER
    if mode == OBLIVIOUS:
        return OBLIVIOUS
    if mode.startswith("FedSDA_"):
        return FED_SDA
    raise KeyError(f"Unknown mode: {mode}")


def selections_for_mode(mode):
    """mode名に埋め込まれたserver flow・検出器・routingを展開する。"""
    method_id = method_for_mode(mode)
    selections = {"method": method_id}
    if method_id == FED_SDA:
        selections["server_flow"] = "NoCached" if "_NoCached_" in mode else "Cached"
        detector_names = (
            "ClassADWIN", "ClassESR", "ClassHDDMA",
            "ADWIN", "ESR", "HDDMA", "HDDMW",
        )
        selections["detector"] = next(
            detector for detector in detector_names if f"_{detector}" in mode
        )
        if mode.endswith("_RestartingSoftRouting"):
            selections["routing"] = "restarting_soft"
        elif mode.endswith("_ProtectedSoftRouting"):
            selections["routing"] = "protected_soft"
        else:
            selections["routing"] = "hard"
        if "_ResidualAdapter_" in mode:
            selections["model_architecture"] = "residual_adapter"
        elif "_SharedBackbone_" in mode:
            selections["model_architecture"] = "shared_backbone"
        else:
            selections["model_architecture"] = "independent"
    elif method_id == WITHOUT_SERVER:
        selections.update({"detector": "ADWIN", "routing": "hard"})
    elif method_id == FED_DRIFT:
        selections["routing"] = "hard"
    return selections


def explicit_option_ids(argv, aliases=None):
    """argvに明示されたCLI名を正規オプションIDへ変換する。"""
    cli_to_id = {
        item.cli_name: item.id for item in OPTIONS if item.cli_name is not None
    }
    cli_to_id.update(aliases or {})
    selected = []
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0]
        if name.startswith("no-") and name[3:] in cli_to_id:
            name = name[3:]
        option_id = cli_to_id.get(name)
        if option_id is not None and option_id not in selected:
            selected.append(option_id)
    return tuple(selected)


def explicit_cli_names(argv):
    """argvに明示された長形式CLI名を返す。"""
    names = []
    for token in argv:
        if token.startswith("--"):
            name = token[2:].split("=", 1)[0]
            if name not in names:
                names.append(name)
    return tuple(names)


def validate_sweep_dependencies(argv, controller_values):
    """無効化された掃引へ固定側オプションが指定されていないか検証する。"""
    explicit = set(explicit_cli_names(argv))
    issues = []
    for dependency in SWEEP_DEPENDENCIES:
        if (dependency.dependent_cli in explicit
                and not controller_values.get(dependency.controller_cli)):
            issues.append(
                f"--{dependency.dependent_cli} requires a non-empty "
                f"--{dependency.controller_cli}"
            )
    return tuple(issues)


def validate_explicit_options(modes, selections, explicit_ids):
    """明示オプションが少なくとも一つの対象modeで実装済みか検証する。"""
    issues = []
    for option_id in explicit_ids:
        per_mode = []
        for mode in modes:
            mode_selections = selections_for_mode(mode)
            mode_selections.update(selections)
            # 数値オプションは依存判定に値を使わなくても、明示された事実を検証対象に含める。
            mode_selections.setdefault(option_id, selections.get(option_id))
            method_id = method_for_mode(mode)
            option_issues = validate_selection(
                method_id,
                mode_selections,
            )
            relevant = tuple(
                issue for issue in option_issues
                if issue.startswith(option_id) or f"{option_id}=" in issue
            )
            if not relevant:
                break
            per_mode.append((mode, relevant))
        else:
            detail = "; ".join(
                f"{mode}: {', '.join(mode_issues)}"
                for mode, mode_issues in per_mode
            )
            issues.append(f"--{option(option_id).cli_name or option_id}: {detail}")
    return tuple(issues)


def render_mermaid():
    """OptionSpecから人間向け依存構造図を生成する。"""
    lines = ["```mermaid", "flowchart LR"]
    categories = []
    for item in OPTIONS:
        if item.category not in categories:
            categories.append(item.category)
    for category in categories:
        lines.append(f"  subgraph group_{category}[{category}]")
        for item in (item for item in OPTIONS if item.category == category):
            choices = " | ".join(item.choices)
            label = f"<b>{item.title}</b><br/>{choices}"
            lines.append(f'    {item.id}["{label}"]')
        lines.append("  end")
    for item in OPTIONS:
        for rule in item.active_when:
            lines.append(
                f'  {rule.option_id} -->|"{rule.label}"| {item.id}'
            )
    for constraint in CHOICE_CONSTRAINTS:
        values = ", ".join(constraint.values)
        for rule in constraint.requires_selections:
            lines.append(
                f'  {rule.option_id} -.->|"{values}: {rule.label}"| {constraint.option_id}'
            )
    lines.extend([
        "  method --> server_flow",
        "  method --> detector",
        "  method --> routing",
        "  method --> new_model_creation_policy",
        "  method --> clustering_policy",
        "  method --> feddrift_detection_batch_size",
        "```",
    ])
    return "\n".join(lines)


def render_sweep_mermaid():
    lines = ["```mermaid", "flowchart LR"]
    for item in SWEEP_DEPENDENCIES:
        controller = item.controller_cli.replace("-", "_")
        dependent = item.dependent_cli.replace("-", "_")
        lines.append(f'  {controller}["--{item.controller_cli}<br/>1個以上の掃引値"]')
        lines.append(f'  {dependent}["--{item.dependent_cli}<br/>固定値"]')
        lines.append(f"  {controller} -->|non-empty| {dependent}")
    lines.append("```")
    return "\n".join(lines)


def render_option_document():
    """依存図と適用状態表を同じスキーマから生成する。"""
    lines = [
        "# 実験オプションの依存構造",
        "",
        "この文書は`experiment_spec/options.py`から自動生成する。直接編集せず、",
        "`python -m tools.generate_option_docs`で更新する。",
        "対象はアルゴリズムの挙動と再現性管理を変えるオプションであり、出力先・seed・再描画などの",
        "単純な入出力指定は含めない。数値パラメータのコード・CLI・論文記号の対応は",
        "`experiment_spec/parameters.py`を正本とする。",
        "",
        "## 読み方",
        "",
        "- **実装済み**: 現在のコードとCLI/modeで選択可能。",
        "- **理論上のみ**: 手法の性質上は移植可能だが、現在の処理フローには未実装。",
        "- **対象外**: 現時点で適用対象としていない。",
        "- `configuration_surface=mode`は、独立CLIではなくモード名へ組み込まれている。",
        "",
        "## 依存構造",
        "",
        render_mermaid(),
        "",
        "## 適用・実装状態",
        "",
        "| オプション | 設定面 | FedSDA | FedDrift | FedSDA_without_server | Oblivious | 説明 |",
        "|---|---|---|---|---|---|---|",
    ]
    labels = {
        "implemented": "実装済み", "theoretical": "理論上のみ", "unavailable": "対象外",
    }
    for item in OPTIONS:
        surface = item.configuration_surface
        if item.cli_name:
            surface = f"{surface}: `--{item.cli_name}`"
        statuses = [labels[implementation_status(item.id, method)] for method in (
            FED_SDA, FED_DRIFT, WITHOUT_SERVER, OBLIVIOUS,
        )]
        lines.append(
            f"| `{item.id}` | {surface} | " + " | ".join(statuses) +
            f" | {item.description} |"
        )
    lines.extend([
        "",
        "## 重要な依存関係",
        "",
        "- `new_model_validation_fraction`は`new_model_creation_policy=validated`でのみ有効。",
        "- `new_model_forward_validation_samples`はforward系またはshadow tournamentでのみ有効。",
        "- forward検証は検出器には依存しないが、現在はFedSDAの仮モデル処理にのみ実装済み。",
        "- SoftRoutingは原理上は検出器から独立しているが、現在の実装はNoCached ClassESRのmodeに限定。",
        "- `server_flow`、`detector`、`routing`は現在独立CLIではなくmode名で組み合わされる。",
        "",
        "## 選択肢固有の実装制約",
        "",
        "| オプション値 | 実装済みmode | 追加前提 | 備考 |",
        "|---|---|---|---|",
    ])
    for item in CHOICE_CONSTRAINTS:
        value = ", ".join(f"`{choice}`" for choice in item.values)
        modes = "<br/>".join(f"`{mode}`" for mode in item.implemented_modes)
        requirements = "<br/>".join(rule.label for rule in item.requires_selections) or "なし"
        lines.append(f"| `{item.option_id}` = {value} | {modes} | {requirements} | {item.note} |")
    lines.append("")
    lines.extend([
        "## 掃引オプションの依存構造",
        "",
        "掃引は対応する`--no-*-sweep`で無効化する。対応する固定値も不要となり、",
        "無効化した掃引へ固定値だけを明示した場合はCLIエラーとする。",
        "",
        render_sweep_mermaid(),
        "",
        "## 集合オプションの既定値と無効化",
        "",
        "| 対象 | 値指定 | 省略時 | 正式な無効化 | 従来の空指定 |",
        "|---|---|---|---|---|",
    ])
    for item in COLLECTION_CONTROLS:
        legacy = "当面受理" if item.legacy_empty_supported else "不可"
        lines.append(
            f"| `{item.id}` | `--{item.values_cli} ...` | {item.default_behavior} | "
            f"`--{item.disable_cli}` | {legacy} |"
        )
    lines.append("")
    return "\n".join(lines)
