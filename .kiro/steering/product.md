# 研究実装の目的

連合学習の概念ドリフトに対するFedSDAとFedDriftを比較する研究実装。
最終提案はResidual Adapter＋ClassESR＋Switching構成で、既存実験は主に完了している。

## 主要な能力

- 時間とクライアントで異なる概念を持つstreamでの比較。
- 検出、routing、割当、候補検証、学習、統合の方式を組み合わせる。
- 精度・回復・通信・計算・モデル数を評価し、条件と来歴を保存する。

## 今回の目的

新API・機能別構成へ移行し、選択肢の追加・削除を追いやすくする。
旧実装は固定した参照。新実装に旧名・旧形式への後方互換を持たせない。
アルゴリズムの変更は構造整理と分けて判断する。

正本: `docs/overview/proposed-method.md`、`docs/research/refactoring-policy.md`。
