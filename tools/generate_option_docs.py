"""experiment_spec/options.pyからオプション依存文書を再生成する。"""

from pathlib import Path

from federated_drift_experiment.experiment_spec.options import render_option_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "options.md"


def main():
    OUTPUT.write_text(render_option_document(), encoding="utf-8")
    print(f"Generated {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
