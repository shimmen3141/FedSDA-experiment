# 生存状態を維持する同期末回収の試作検証

通常経路への採用後は[通常FedSDAの同期末回収](fedsda-runtime-memory-reclamation.md)を参照する。この試作は当時の固定source専用で、更新後のsourceでは開始前に停止する。

## 目的と範囲

uraniumで観測した数GiB/runの使用量に対し、モデル・学習標本・検証/診断記録を保持したまま、同期の正常終了後に解放済み領域をOSへ返す試作を検証する。学習ロジック・データ保持数・既定のallocator設定は変更しない。

`tools/research_memory_reclamation/run.py`だけから起動する独立した研究CLI。製品src、通常pytest/CI、構成オプションや標準JSON/NPZ形式への登録はない。専用checksは`checks.py`の明示実行のみ。削減効果・採用の判断はサーバ結果を見て人が行う。

## 実行する2run

共通条件はFedSDA・MNIST2・seed0・10client・5000標本/client・間隔50・1worker。独立した子processを**control→reclaim**の順に実行し、並行しない。数値threadは各1に固定し、既存のallocator環境変数を保存して両runで維持する。

- control: 同期末のRSS/high-waterを読むだけ。
- reclaim: 同じ位置で`malloc_trim(0)`を1回呼び、その直前/直後RSSと所要時間・返却結果を記録する。戻り0も有効な記録として扱う。

追加の監視thread、周期的PSS読取り、強制GC、学習データ削除は行わない。元CLIの保存処理を使い、run返却時のhigh-waterをhash生成より先に記録する。GNU timeの最大RSSには初期準備・hash生成・保存も含まれるので、run返却までの値と分けて比較する。

最後に完全設定・全指標・全配列のキー/shape/dtype/bytes・最終サーバパラメータhashを照合する。許容誤差は広げない。失敗・コード/環境変更・未完了・不一致なら非成功で終了し、元記録を保持する。既存の出力先へ再開/上書きしない。

## uraniumでユーザーが行うこと

転送用ZIPは`results/fedsda-memory-diagnosis/uranium-live-reclamation-20261019.zip`。このZIPだけをリポジトリへ渡して実行でき、git pullやライブラリ更新は不要。展開先はgit管理外の`results/fedsda-memory-diagnosis/reclamation-prototype-20261019/`で、過去の測定結果を変更しない。

### Windows PowerShell: 踏み台経由で転送

接続先2つを、普段使っている値かSSHエイリアスへ置き換える。uraniumのホスト名は踏み台から到達できるものを指定する。

```powershell
$jumpTarget = '踏み台のユーザー名@踏み台のホスト名'
$uraniumTarget = 'uraniumのユーザー名@uraniumのホスト名'
$archive = 'C:\Users\yshin\Research\FedSDA-experiment\.worktrees\refactoring\results\fedsda-memory-diagnosis\uranium-live-reclamation-20261019.zip'
scp -J $jumpTarget $archive "${uraniumTarget}:uranium-live-reclamation-20261019.zip"
ssh -J $jumpTarget $uraniumTarget
```

踏み台のportが22以外なら`$jumpTarget = 'ユーザー名@ホスト名:ポート番号'`とする。

### uranium: tmux内で実行

`cd`のパスを実際のリポジトリへ置き換える。別の重い実験との同時実行を避ける。2runで概ね30〜40分を目安とするが、試作のコスト・サーバ負荷により変わる。

```bash
cd /実際のパス/FedSDA-experiment
tmux new -s live-reclamation
```

tmux内で実行する。

```bash
source .venv/bin/activate
python -m zipfile -e ~/uranium-live-reclamation-20261019.zip .
python results/fedsda-memory-diagnosis/reclamation-prototype-20261019/run.py \
  --purpose 'モデルと標本を保持した同期末回収のピークRSS・時間・全保存値を、回収なしと比較する。採否は自動決定しない。' \
  --output-directory "results/fedsda-memory-diagnosis/live-reclamation-$(date +%Y%m%d_%H%M%S)"
```

引数を追加しなければ5000/clientで実行する。`--sample-count 50`は手元の接続確認専用で、本測定には指定しない。ソースhashやLinux/glibc/GNU timeが基準と合わないときは学習前に停止する。値を合わせるためにコードやライブラリを変更せず、エラーを担当者へ渡す。

切断するときはCtrl+Bを離してD。再接続後は`tmux attach -t live-reclamation`で戻る。

### Windows PowerShell: 結果を返す

実行時に表示される`live-reclamation_<日時>`ディレクトリ全体を、成功・失敗にかかわらず返す。日時の部分は実際の出力名に置き換える。別のPowerShellなら接続先変数も設定し直す。

```powershell
$remoteResult = '/実際のパス/FedSDA-experiment/results/fedsda-memory-diagnosis/live-reclamation-日時'
$localResults = 'C:\Users\yshin\Research\FedSDA-experiment\.worktrees\refactoring\results\fedsda-memory-diagnosis'
scp -r -J $jumpTarget "${uraniumTarget}:${remoteResult}" $localResults
```

## 成果物の読み方

- `plan.json`: 条件・目的・コードidentity。
- `control/`と`reclaim/`: 元CLIの`cli/`、同期末の`events.jsonl`、最終`parameters.json`、前後のenvironment、`gnu-time.txt`、stdout.log。
- `comparison.json`: 全保存値が一致した場合の`status=passed`と、各armのピーク/時間/生存中回収量。`passed`は採用決定ではない。
- `failure.json`または完了記録の欠落: 実行未成功。ログを保持し、条件を変えて元結果を上書きしない。

手元の検証・独立レビューの正本は`.kiro/specs/run-memory-reclamation-prototype/{spec.json,tasks.md}`。uraniumの効果測定はユーザーからの結果返却後に記録する。

## 手元で完了した検証

Linux/WSL・Python3.14.4・NumPy2.4.6・torch2.12.1+cpuの1000/client・MNIST2・seed0・20ラウンドでは、対照間の完全設定・全指標・全133配列bytes・最終パラメータhash・環境が一致した。

| 項目 | control | reclaim |
| --- | ---: | ---: |
| GNU time最大RSS MiB | 1173.98 | 1114.03 |
| wall秒 | 145.16 | 152.00 |
| 同期末回収の呼出し | 0 | 20 |
| 最大の同期末RSS低下 MiB | 0.00 | 440.88 |

観測されたピーク差は約59.95 MiB（約5.1%）、wall差は+6.84秒（約4.7%）。回収関数の累積呼出し時間は約0.47秒。1対の観測なので普遍的な削減率/性能悪化率とは扱わず、uraniumの5〜9 GiB条件に一般化しない。結果は`results/fedsda-memory-diagnosis/reclamation-local1000/`。

その前の50/client・1ラウンドの接続確認も全保存値一致。ただし整形前runnerの記録で、独自のidentityを保持している。独立レビュー後に133配列の両側同時欠落を拒否する照合ゲートを強化し、最終配布runnerでも50/clientの2runと全保存値一致を再確認した。専用checksは最終24件、公開名照合、Python3.10文法、isolated Ruff/formatも主担当が確認。製品コード・共通pytest/CIは未変更で、全pytestは再実行していない。
