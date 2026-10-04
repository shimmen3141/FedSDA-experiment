# 調査・設計判断
## Summary
旧servers/base.py:130–183とshared_backbone.py:54–130を直接確認。参加選別済み全体momentsの件数加重平均だけを切り出す。
## Research Log
client登録順、model所有＋正のtraining件数が参加gate。stats欠落は寄与なし。重みはstats.nでありtraining件数ではない。
通常のmean*n +=/n +=順を維持、正n時M2=0のwhole統計を作りclass情報を持たない。zero合計では既存globalrecordを変更しない。
Baseと共有バックボーンでmodel/outerloop/aggweightsの意味は異なるが、unique activeIDsの各モデルの損失平均は同じ。
モデルを生成しないSimpleNamespace get_params stubで旧Base全体methodを実測。training100/1、n1/3、mean.1/.9からn4/mean0.7000000000000001/M2=0を確認。
class情報・モデルID/順序・参加gate/既存保存は上位で明示し、このpure入力には取り込まない。

## 実測した極大件数
n=(2**53,3,3)、全mean1では旧平均1.0000000000000002を登録する。n=(10**308,10**308)、全mean1では旧除算がOverflowError、先行model更新は既に行われる。
通常client/過去成果への影響は未確認。旧productionを修正しない。結果の既存型契約を満たさない極大入力は新pure関数で補正せず拒否する。
詳細の正本は../../../docs/research/implementation-findings/legacy-008-extreme-count-server-loss-aggregation.md。後続testで旧/新同入力を対照する。

## Synthesis
既存momentsを入力/出力に使い、Noneを上位の既存保持signalとする。公開constructorで全入力先行検査、算術/結果検査。
分散pooling・新records・汎用aggregationframework不要。数値ライブラリ不要。
cc-sddの要件/設計保存前gate、fable-methodの実旧oracle・先行極端条件確認を適用。日本語文書はUTF-8直接patchと保存後読み返しを使う。
