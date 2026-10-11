# Requirements Document

## Project Description

FedSDAとFedDriftを比較する研究者のため、旧FedDriftのサーバ進行を新構成へ移植する。現状はclient組立まで完了し、サーバとclientが保存済みモデルをそれぞれ平均する数値部が未移植である。歴史的な重み・演算順・統計のリセットを保った独立の計算へ移し、後続のサーバ進行から利用できるようにする。

## Requirements

### 1 保存済みモデルの加重統合

1.1 When 保存済みモデルの分割が指定される, the FedDrift consolidation shall 各群の最小IDを代表とし、群と群内モデルの指定順を保つ。
1.2 When 過去のモデル別重みが与えられる, the FedDrift consolidation shall 欠落を0とし負値を0へ切り上げ、群内合計が0なら全モデルを重み1とする。
1.3 When 群を統合する, the FedDrift consolidation shall 各モデルのパラメータに重みを合計で割った比率を掛け、指定順に加算して旧と完全一致する値を返す。単独群もこの演算を行う。

### 2 統計・隔離期限とclient側の計算

2.1 When サーバ側の群を統合する, the FedDrift consolidation shall 存在するモデルの損失件数の合計と件数加重平均を返し、偏差平方和を0へ戻す。合計0件なら平均も0とする。
2.2 When サーバ側の群を統合する, the FedDrift consolidation shall 統合に使用した重みの合計を保存用重みとし、群内の最大隔離期限が現在時刻より後の場合だけ代表の期限として返す。
2.3 When client側にサーバが決めた群内重みが渡される, the FedDrift consolidation shall client自身の保存済みパラメータを1.3と同じ演算で統合する。重み合計0の場合は均等重みとする。

### 3 計算の分離と入力の拒否

3.1 The FedDrift consolidation shall 入力の値・参照先・勾配・乱数状態を変更せず、入力と共有しない出力パラメータを返す。空のモデル集合と空の分割には空の結果を返す。
3.2 If 分割・モデルID・パラメータの構造/型/寸法/有限性・重み/期限・損失統計が契約を満たさない, the FedDrift consolidation shall 入力を変更せず例外で拒否する。
3.3 The FedDrift consolidation shall 新構成内の計算として動作し、旧実装を実行時依存に含めない。固定旧実装・goldenの値を変更しない。

## Scope

本specは数値一致が繊細な加重統合の計算だけを切り出す（共通手順の例外）。クラスタを決めるクロス評価/距離/linkage、clientのownerへの適用・optimizer再構築・標本storeの再配置、通信記録、隔離モデル登録、FedAvg、R回進行は後続サーバspecが扱う。統合結果を新全体runへ接続した、または全体runのgolden一致を達成したとは扱わない。
