"""FedSDAの多因子ablation用study manifestをUTF-8定義から生成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from federated_drift_experiment.experiment_spec.manifests import write_json_atomic
from federated_drift_experiment.experiment_spec.parameters import (
    PARAMETER_SCHEMA_VERSION,
)


STUDY_SCHEMA_VERSION = 1


def build_study_manifest(definition_path, study_root, check=False):
    definition_path = Path(definition_path)
    study_root = Path(study_root)
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    variant_definitions = definition["variants"]
    reference = definition["reference_variant"]
    if reference not in variant_definitions:
        raise ValueError(f"reference_variantがvariantsにありません: {reference}")

    variants = {}
    for variant_id, variant_definition in variant_definitions.items():
        relative_path = Path(variant_definition["path"])
        variant_manifest_path = study_root / relative_path / "manifest.json"
        if not variant_manifest_path.is_file():
            raise FileNotFoundError(f"variant manifestがありません: {variant_manifest_path}")
        variant_manifest = json.loads(
            variant_manifest_path.read_text(encoding="utf-8")
        )
        if variant_manifest.get("variant_id") != variant_id:
            raise ValueError(
                f"variant_idが一致しません: {variant_manifest_path}"
            )
        selection = variant_manifest.get("selection", {})
        summary = {
            "path": relative_path.as_posix(),
            "mode": variant_manifest["mode"],
        }
        for axis in definition["comparison_axes"]:
            summary[axis] = selection.get(axis)
        summary["included_datasets"] = selection.get("datasets", [])
        summary["missing_datasets"] = selection.get("missing_datasets", [])
        variants[variant_id] = summary

    manifest = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "parameter_schema_version": PARAMETER_SCHEMA_VERSION,
        "study_id": definition["study_id"],
        "title": definition["title"],
        "status": definition.get("status", "ablation_reference"),
        "question": definition["question"],
        "comparison_axes": definition["comparison_axes"],
        "common_configuration": definition["common_configuration"],
        "reference_variant": reference,
        "variants": variants,
    }
    output = study_root / "manifest.json"
    if check:
        if not output.is_file():
            raise FileNotFoundError(f"study manifestがありません: {output}")
        current = json.loads(output.read_text(encoding="utf-8"))
        if current != manifest:
            raise ValueError(f"study manifestが定義と一致しません: {output}")
    else:
        write_json_atomic(output, manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = build_study_manifest(
        args.definition, args.study_root, check=args.check,
    )
    action = "checked" if args.check else "saved"
    print(f"Study manifest {action}: {manifest['study_id']}")


if __name__ == "__main__":
    main()
