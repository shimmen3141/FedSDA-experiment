# Implementation Plan

- [ ] 1. 検出器と監視
- [ ] 1.1 ADWINの検出器
  - 旧の検出器と照合するtestを先に書く。
  - _Requirements: 1.1_
- [ ] 1.2 ADWINでの監視と、補助の判定
  - _Depends: 1.1_
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 2. 構成の追加
- [ ] 2.1 2つの構成の名前と、実旧との照合
  - _Depends: 1.2_
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 3. 検証
- [ ] 3.1 独立レビューの指摘を反映し、全回帰を通す
  - _Depends: 2.1_

## Implementation Notes

（実装中の発見、レビューの指摘の採否、検証結果、未検証と残る制約を書く）
