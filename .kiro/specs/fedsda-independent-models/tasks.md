# Implementation Plan

- [ ] 1. モデルと学習
- [ ] 1.1 共有部を持たないモデル構造と、空の共有部のoptimizer
  - 実旧との照合のtestを先に書く。
  - _Requirements: 1.1, 1.2, 1.3_
- [ ] 1.2 モデルごとに順に学習する方式と、設定の組
  - _Depends: 1.1_
  - _Requirements: 2.1, 2.2_

- [ ] 2. サーバと構成
- [ ] 2.1 空の共有部の集約と、クロス評価の通信量
  - _Depends: 1.1_
  - _Requirements: 3.1, 3.2_
- [ ] 2.2 構成の名前と、実旧の全体runとの照合（sine2とmnist2）
  - _Depends: 1.2, 2.1_
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 3. 検証
- [ ] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - _Depends: 2.2_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
