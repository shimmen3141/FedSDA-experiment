"""実験の掃引のコマンドの入口。使い方: `python run_experiment_sweep.py --help`（docs/experiments/experiment-sweep.md）。

packageのinstallや、環境変数の設定なしに実行できるように、この場所の`src`を、読込みの経路へ足す。
"""

import os
import sys
from pathlib import Path

# 数値計算のライブラリ（OpenMP・MKL・OpenBLAS）のthreadを、processごとに1つにする（NumPyとtorchを読み込む前に決める）。
# 並列のworkerのprocessは、このファイルを読み直すので、同じ設定になる。すでに設定されている値は、変えない。
for thread_count_variable_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(thread_count_variable_name, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    # workerのprocessでは、上の、環境変数と経路の設定だけが実行される。
    from federated_learning_experiments.sweep.experiment_sweep_command import main

    raise SystemExit(main())
