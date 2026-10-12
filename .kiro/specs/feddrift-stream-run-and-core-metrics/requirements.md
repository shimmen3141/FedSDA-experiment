# Requirements Document

## Introduction
FedDriftを設定から事前学習・全標本生成・全chunk観測・通常同期・終端まで実行し、新しいFedSDAと共通の定義による基本指標を得る。既存の共通評価関数と記録型を使い、FedSDA専用の候補検証/予測混合/再較正の項目を無理に埋めない。計算量・変更応答/モデル/予測相補性の追加診断、CSV/NPZ/CLI接続は後続。固定旧・既存goldenは変えない。

## Requirements
### Requirement 1: 設定から全体runを実行する
1.1 When 有効な設定とfactoryを与えた時, the FedDrift全体run shall 同じrun乱数で事前学習→全client概念列→全標本列を生成し、client順の完全な検出chunk観測→serverの時刻準備→設定されたR回の「client順局所学習→server学習round」を繰返し、既存の観測済みストリーム終端を実行する。
1.2 The FedDrift全体run shall 全標本を生成しつつfloor(標本数/chunk)個だけ処理し、未処理末尾と全概念列を共通run結果へ残す。chunk未満のrunも準備/生成/終端のみで成功する。
1.3 The FedDrift全体run shall 呼出元のPython/NumPy/CPU torch乱数を成功・失敗とも変えず、同じ設定で再実行した値・状態を再現する。
### Requirement 2: 入力と失敗境界
2.1 If 実行設定、factory設定、seed範囲またはchunkと検出件数の対応が不正な場合, then the FedDrift全体run shall 参加者生成・乱数消費より前に拒否して直前のprepared集合を保持する。
2.2 If 準備/生成/観測/同期/終端が失敗した場合, then the FedDrift全体run shall 段階と可能なclient/標本/round位置、元causeを保持する共通実行例外を返し、成功していない段階を成功イベントへ記録しない。途中の参加者状態のrollbackは提供しない。
### Requirement 3: 共通の基本指標
3.1 When run結果と準備済み参加者と共通の指標設定を与えた時, the FedDrift指標導出 shall 共通計算で精度・定常精度・学習モデル切替の検出指標を返し、最終モデル数・通信量・全モデルのparameter値数/bytesを新FedSDAと同じ名前と単位で返す。全概念列の未処理末尾の変更も見逃しへ含める。
3.2 The FedDrift指標導出 shall 状態・3乱数を変えず、不正な型/設定、client対応、生成/処理件数、予測記録との不整合を拒否する。空の予測では精度0・定常精度NaNとする。
### Requirement 4: 全体runの一致と独立起動
4.1 The 検証 shall 実旧run_random_drift_experimentを実行し、全標本/概念列・全client/server状態・通信・3乱数と基本指標を完全に照合する。
4.2 When Linuxで固定旧を別processで2回実行して再現した時, the 検証 shall 別のLinux goldenを保存し、新全体runの基本指標/最終parameter hashを照合する。既存Windows goldenとtestは変更しない。
4.3 The 全体runと指標導出 shall 共用freshで旧/test importなしに接続でき、exact依存境界を維持する。
