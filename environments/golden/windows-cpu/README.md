# Windows CPU goldenの再現環境

既存11ケースと最終Residual Adapter＋Switchingの3ケースが一致したローカル環境を保存する。
研究室サーバの論文用実験環境は別に扱う。

## 保存内容

- `requirements-freeze.txt`: 間接依存・pipを含む27パッケージの版固定。
- `environment.json`: OSビルド、CPython、CPU型番・命令セット、dtype、スレッド設定、
  NumPy/OpenBLAS・torch/MKLのビルド情報、ソースcommit、goldenのSHA-256。
- `verification.json`: 新規venvへの再構築後の環境照合・回帰テスト結果。
- `recreate.ps1`: 新規venvへ版固定した依存をインストールする。
- `capture.py`: 指定先へ実行環境を記録する。
- `verify.py`: 保存環境と照合し、スキーマ・関連機能・旧/新goldenを検証する。

ここで保存するのは、2026-10-02に**既存goldenと一致した環境**である。
Pythonの版が異なる旧goldenの元生成環境を復元した、という意味ではない。
golden JSON内の生成時メタデータは書き換えない。

2026-10-02に新規venvへ公式PyPIから再構築し、27パッケージと数値計算ビルドの一致、
`pip check`成功、旧/新goldenを含む**113テスト成功（skipなし）**を確認した。
詳細は`verification.json`に保存している。元のvenvとgoldenの内容は変更していない。

## 前提

- Windows x86-64、CPython **3.13.15 / 64bit**。
- Python公式配布のWindowsインストーラで用意する。venvはPython本体を保存しない。
- 基準PCはAMD Ryzen 7 8845HS、Windows 11 build 26200。
- 再構築では公式PyPI `https://pypi.org/simple`を使用する。
- MNIST学習用の画像・ラベルを`data/mnist/`へ配置するか、`FDE_MNIST_DATA_DIR`を指定する。
  初回取得にはインターネット接続が必要。データのSHA-256は最終構成goldenに記録している。

torchの配布パッケージ版は`2.12.1`、実行時表示は`2.12.1+cpu`で、CUDAは無効。
この組合せをPyPIのWindows wheelから再構築する。別のCPU indexのwheelへ置き換えない。

## 再構築と検証

リポジトリのルートでPowerShellから実行する。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File environments/golden/windows-cpu/recreate.ps1
.\venv\golden-rebuild\Scripts\python.exe environments/golden/windows-cpu/verify.py
```

既定ではPython Launcherの`py -3.13`を使用し、起動したPythonが3.13.15か確認する。
別のPythonを使う場合は`-PythonExe C:/path/to/python.exe`を指定する。
保存先を変える場合は`-VenvPath venv/golden-rebuild-2`を指定する。
保存先が存在すれば停止するので、既存の環境は上書きしない。
Windows PowerShell 5.1でも日本語を読めるよう、`recreate.ps1`はUTF-8 BOM付きで保存する。

`verify.py`は子プロセスに`OMP_NUM_THREADS=1`・`MKL_NUM_THREADS=1`を渡す。
環境・依存・数値計算ビルドの一致を確認してからテストを実行する。
CPU型番やWindowsビルドの差は報告し、実際の回帰結果で判断する。
依存やgoldenに差があれば停止し、goldenの自動更新は行わない。

検証結果・JUnit・再構築した環境の記録は、対象venvの`golden-verification/`へ保存する（git管理外）。
同じコマンドを後日実行しても、このディレクトリの基準記録は上書きしない。

## 基準の更新

現在の基準を残したまま、新しい保存先へ記録する。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\venv\Scripts\python.exe environments/golden/windows-cpu/capture.py --output-dir venv/environment-candidate
```

意図した依存更新・手法変更を確認し、新規環境で検証してから基準を更新する。
`pip freeze`はインストール済み版の一覧で、wheel内容をハッシュ固定したlockfileではない。
将来の配布終了や別CPU・別OSでも同じ数値が得られることまで保証しない。
再構築に使ったwheelやPythonインストーラも長期保管すれば、配布終了への備えを強められる。
