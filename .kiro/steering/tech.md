# 技術と検証

## 現状

既存実装はPython・NumPy・PyTorch。新src実装はまだない。
Windows CPUの固定環境を別venvから再構築し、113テスト成功を確認済み。
版とビルド情報は`environments/golden/windows-cpu/`を参照する。

## 原則

- 不変の型付き設定を渡し、実行中にグローバル設定を変更しない。
- 依存を組み立てるruntimeと、機能内の計算・判断・状態を分離する。
- 明確な単位・ID・時系列を使い、実際の差し替え境界だけを抽象化する。
- PyTorch依存はモデル・学習の責務として明示する。

## 検証

最終3ケースと移植対象baselineについて、旧goldenの数値・イベント列を比較する。
旧APIの受理で対応せず、テスト内で新しい記録と対応付ける。
乱数消費、反復順、optimizer更新、共有参照、ID採番を維持する。
環境差でgoldenを上書きしない。

Windows基準の手順は`docs/experiments/refactoring-baseline.md`。
長時間研究実験の実行環境はLinuxであり、開発用golden環境と区別する。
