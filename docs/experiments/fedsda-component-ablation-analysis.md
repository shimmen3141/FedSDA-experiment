# 現行FedSDAの構成要素比較と最終評価の手順

2026-10-17。混合予測・FIFO容量30・共通のFixed-Share時間尺度T120を基準にする。
Tはモデル別重みと配布時/ローカル予測の混合で共用し、分離しない。
この文書は分析前の判断手順であり、要素の削除や既定値の変更を決定するものではない。

## 判断の順序

1. 完全設定・環境・来歴・seedの対応を確認する。欠損を埋めたり、異なる時間尺度を混ぜたりしない。
2. 各dataset×集約間隔で、全体精度と安定期精度、同seedの差、悪化するseedを確認する。
3. 回復期精度と検出指標で、悪化が定常中か変化後かを調べる。スイッチ遅延は予測回復時間と区別する。
4. 上り/下り通信総量と標本当たり通信、計算総量と標本当たり計算、モデル数を併せて確認する。
5. 複雑さを減らす利点と条件別の悪化を人が判断し、採否・理由・例外・追加検証を記録する。

**許容する精度低下幅は未設定で、自動採否は行わない**（ユーザー決定）。
「ばらつきの範囲だから同等」「有意差がないから不要」とは判断しない。
5seedの平均とSEは探索の材料であり、非劣性の証明ではない。
全datasetの総平均で局所的な悪化を相殺しない。指標別の欠測とrun自体の不足を分ける。
精度と通信量が交換関係なら、勝者を自動選択せず、悪化条件と資源削減量を残す。

## 今回の比較と解釈できる範囲

各構成は6dataset×seed 0–4×間隔50/100/200/500の120run、計720run。
client数10・各client標本数5000。2026-10-17に転送された結果で全720run完走と720対・17指標の比較成立を確認した。実在保存先・来歴・数値と解釈は[結果の記録](../research/fedsda-component-ablation-20261017.md)を参照。要素の採否と既定の変更は行っていない。

| 構成名 | 主な問い | 注意 |
|---|---|---|
| `fedsda-without-candidate-validation` | 新規モデルの将来標本での検証が必要か | 精度だけでなく候補増加に伴う通信・計算・モデル数を見る |
| `fedsda-without-prediction-weight-recalibration` | 集約後のFIFO再較正が混合予測でも寄与するか | 旧局所予測・T30での結論をそのまま持ち込まない |
| `fedsda-loss-distance-clustering` | クラス別機能比較と損失距離のどちらを使うか | クラスタリングを丸ごと無効にする比較ではない |
| `fedsda-overall-loss-esr-detection` | クラス別監視を含める必要があるか | e-SRは共通。ADWIN置換とは異なる |
| `fedsda-without-concept-specific-adapter` | 概念別adapterの寄与は何か | 共有部は残る |
| `fedsda-independent-models` | 共有構成と独立モデル構成の交換関係は何か | 共有部だけでなく逐次モデル別学習と再較正なしも変わる |

独立モデルと再較正なし構成の差も補助的に確認できるが、学習方式の差が残るため
「共有バックボーンだけの因果効果」とは書かない。現行CLIは両構成をそれぞれ基準と比較する。

## 基準結果の再利用

基準は次の実験のT120部分だけを使う。

```text
results/fedsda-prediction-timing/
  20261016_142436_synchronized-local-mixture-fifo30-time-scale120-240-n5000/
    fedsda-synchronized-local-mixture-prediction/runs/
```

保存された基準版は5、commitは`3e6153ce594050957e42910ffcef7b2bfcb68590`。
T120の全120runの完全設定は、`6c3522d`時点の`fedsda`と一致する。
両commit間のsrc差分は混合/T120の既定化、局所予測構成と単一モデル予測の明示、基準版番号で、
学習アルゴリズムの変更はないことを調査した。
この調査は**任意の後続commitの同等性を保証しない**。サーバのrevisionが異なれば差分も確認する。
基準の環境記録はLinux/x86_64、Python3.10.16、NumPy2.2.6、torch2.12.1+cpu。
変更側と一致することをCLIが確認する。保存項目が一致してもCPU型番やbackendまで保証するものではない。
既存runの来歴を基準版6に書き換えたり、今回の実験ディレクトリへ複製したりしない。

## 分析コマンド

研究室サーバで、結果と同じcheckoutを使う。`ABLATION_RUN`は実在する今回の保存先に置き換える。
分析のための実験再実行は不要。分析CLIは学習を開始しない。

```bash
source .venv/bin/activate
ABLATION_RUN='results/fedsda-component-ablation/20261017_041328_core-components-mixture-fifo30-t120-n5000'
python tools/research_component_ablation/analyze.py \
  --baseline-directory results/fedsda-prediction-timing/20261016_142436_synchronized-local-mixture-fifo30-time-scale120-240-n5000/fedsda-synchronized-local-mixture-prediction \
  --ablation-directory "$ABLATION_RUN" \
  --configurations \
    fedsda-without-candidate-validation \
    fedsda-without-prediction-weight-recalibration \
    fedsda-loss-distance-clustering \
    fedsda-overall-loss-esr-detection \
    fedsda-without-concept-specific-adapter \
    fedsda-independent-models \
  --datasets sine2 sea2 sea4 circle2 mnist2 mnist4 \
  --seeds 0 1 2 3 4 \
  --aggregation-intervals 50 100 200 500 \
  --client-count 10 \
  --per-client-sample-count 5000 \
  --allow-provenance-difference
```

最後のオプションは、上記の来歴差を確認したうえで指定する。
コードの同等性を自動証明したり、環境差やdirtyなrunを許可したりするものではない。
今回のサーバのcommitが上記調査に含まれることを先に確認する。
来歴が同じ場合はこのオプションを外せる。

出力は`$ABLATION_RUN/analysis_<UTC日時>/report.md`と`paired.csv`。
同じ入力でも新規ディレクトリを作り、既存結果を上書きしない。
`--stdout`を加えるとMarkdownを表示するだけでファイルを作らない。
exit0は期待run格子の全対が比較可能、exit1は不足/比較不能、exit2は引数・入力ルートの誤り。
指標の未定義だけではexit1にならないため、exit0でも指標別の有効対数を確認する。
欠損runの原因やNPZ内部の完全性は、この集計だけでは確認できない。
現在のbuilderとの完全設定一致を要求するため、将来既定が変わった場合は解析コードのrevisionも固定して使う。

台帳へ残すのは実在する保存先、sweep_recordのpurpose/command/code revision/基準版/環境、
予定720件とcompleted/skipped/failed件数、分析出力パス、採否の判断である。
欠損があれば、まずsweep_recordとログを調べる。完走した結果を消して再実験しない。

## 構成選択と最終評価を分ける

今回のseed 0–4は構成選択用。予測方式やTの探索で使ったseed 0–9も最終評価の未使用seedとは呼ばない。
採否後に構成・比較対象・指標・計算/通信の算定方法を固定し、未使用であることを台帳全体で確認したseed集合を
最終比較に割り当てる。候補は10–19だが、未使用確認前には確定しない。
最終結果を見て再調整したら、その集合も探索用とし、別の評価集合を用意する。

同じ標本列・概念スケジュール・seed・集約間隔・学習予算でFedDrift、FedAvg、採用した追加手法と比較する。
標本の予測を正解の観測/学習より先に行う時系列、ラベル到着条件、client参加条件を揃える。
単純なFedAvg比較と、複数概念に対応するFedDrift比較は役割を分けて残す。
追加手法は[主軸と比較対象の整理](../research/method-positioning.md#8-主張と必要な証拠の対応2026-10-17)を参照。

## 共通検証からの分離（2026-10-18）

解析CLIと専用検証は`tools/research_component_ablation/`内に置く。通常のpytest・共有Ruff設定には登録せず、研究用の検証は次のように明示実行する。

```bash
python -m pytest tools/research_component_ablation/checks.py -q -p no:cacheprovider
```

旧入口`analyze_fedsda_component_ablation.py`は移設した。解析計算・結果形式は変更していない。過去の分析来歴と成果物は保持し、移設後の実行からscript hashとコマンドパスが変わる。通常の学習フローからこの解析CLIを呼ぶ経路はない。

### 研究用コードの点検範囲と分離状況

基準版6確定後（6c3522d以後）の追加履歴と、src・通常test・CI・共有設定からの参照を確認した。

|対象|現在の位置・扱い|共通の学習フローへの実験用分岐|
|---|---|---|
|構成要素比較の解析|tools/research_component_ablation。共有Ruff登録と通常pytestの専用26件を撤去|なし|
|待機後強制採用D|専用CLIとtools/research_candidate_validation_waitingの明示検証。前回分離済み|なし|
|初期予測質量P/R|tools/research_prediction_initial_mass内だけ。通常収集/共有登録なし|なし|
|共有部25%反映・候補共有部固定などの一時対照|results配下の研究script。srcからの呼出しなし|なし|

掃引の再開補助（ユーザー依頼で追加した正式機能）は残す。旧新版比較CLIとその通常testは移植検証として以前からあるもので、一時研究介入ではないため今回の移設対象に含めない。採用済み構成やアブレーション選択肢自体も削除していない。
