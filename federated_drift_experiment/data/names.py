"""正規データセット名の検証とCLI公開範囲。"""


CANONICAL_DATASET_NAMES = (
    "blobs", "sea2", "sea4", "circle2", "sine2", "mnist2", "mnist4",
)


def normalize_dataset_name(dataset):
    """正規名を返し、旧名・未知名は暗黙変換せず拒否する。"""
    if dataset is None:
        return None
    name = str(dataset)
    if name not in CANONICAL_DATASET_NAMES:
        choices = ", ".join(CANONICAL_DATASET_NAMES)
        raise ValueError(f"Unknown dataset: {name!r}. Choose one of: {choices}.")
    return name


def dataset_cli_choices(canonical_names):
    """CLIで受理する正規データセット名だけを返す。"""
    return tuple(canonical_names)
