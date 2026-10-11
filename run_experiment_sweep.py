"""実験の掃引のコマンドの入口。使い方: `python run_experiment_sweep.py --help`（docs/experiments/experiment-sweep.md）。

packageのinstallや、環境変数の設定なしに実行できるように、この場所の`src`を、読込みの経路へ足す。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    # 並列のworkerのprocessは、このファイルを読み直す。上の経路の追加だけが、そこで実行される。
    from federated_learning_experiments.sweep.experiment_sweep_command import main

    raise SystemExit(main())
