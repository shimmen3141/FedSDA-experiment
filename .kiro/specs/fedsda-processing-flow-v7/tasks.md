# Implementation Plan

- [x] 1. 将来予測の短縮と説明の読みやすさを整える
  - [x] 1.1 v7専用の期待列/行間/収まり検査を用意し、F01短縮とcaption余白を反映する。18経路472コマの期待列が一致すること。
  - _Requirements: 1, 3_
- [x] 2. モデル状態の表示を実コードへ合わせる
  - [x] 2.1 過渡opacityとcross評価/候補初期学習の検査を用意し、focus切替・独立評価copy・候補/固定参照を反映する。実時間の濃淡と対応する図が根拠に一致すること。
  - _Requirements: 2, 3_
- [x] 3. 表示と操作を検証して独立レビューを受ける
  - [x] 3.1 3画面の全18経路、実時間の遷移、代表PNGを確認し、独立CLI実装レビューと指摘採否/根拠を記録する。旧版を保持しcommit/pushしない。
  - _Requirements: 1, 2, 3_

## Implementation Notes
- 対象はv7とこのspecのみ。v2〜v6と_1など既存の未追跡成果物を削除/stageしない。

- 仕様レビューはLuna mediumでPASS。Minorは既定の退出中過渡検査と候補学習検査の実施確認で、実施して対応した。sourceの数値変更はない。
- 将来標本はfuture:F01の1コマへ短縮し、P01〜P07再演を除いた。captionの見出し/説明間を3px、行高も広げ、狭幅で説明が枠に収まることを全コマ確認した。
- 親actor .28sと子state .25sの逆方向transitionでフラッシュを再現した。子のtransitionを止め、モデル群が非活性の間は直前focusを保持する。親のfadeと矢印アニメーションは保持した。
- クロス評価はサーバ集約後コピーを独立actorで表示。R06の手元モデルは正誤比較の診断用で、共有部も含む。配布時側は使用しない。C01はFIFO変化区間からの候補初期学習、C02は更新用の既存モデル群から固定参照への写しを明示した。
- 根拠（refactoring HEAD c4222ff、読取りのみ）: alarm_training_interval_preparation.py:139からFIFO区間分割、alarm_buffer_response.py:238から変化区間標本を解決処理へ。post_alarm_candidate_validation_session_start.py:238/295/334は候補初期学習→既存モデル群の固定参照作成。model_cross_evaluation.py:205/268は集約後snapshotを渡す。client_model_cross_evaluation.py:241/247は独立classifierへload、278は手元更新用モデルの正誤比較。post_alarm_reference_model_fixation.py:30は既存モデル群の独立copy。

- 最終HTMLを1280×720/390×844/1440×900の各18組合せ・472コマで検査し、issues=0。固定配置/文字の収まり/新規候補の学習経路/独立cross評価copy/短縮列/既存の共有/コピー/末尾循環/モーダル外クリックとEscapeを確認。通常モーションの流れと低減モーションも維持した。
- 実時間検査はtransitionを無効化せず、N01→N02、R12→R03、N05→R12、R10→R11、逆送り、非活性→非活性を確認してPASS。時点はperformance.nowによる実測elapsedも記録した。旧v6ではN01→N02でsnapshot実効opacity .22→.3705575のREDを再現した。
- 将来予測の短縮/候補初期学習/クロス評価/固定参照/狭幅説明の代表PNGを確認した。最終構文検査、実idの重複なし/外部assetなしも確認した。
- 独立CLI実装レビューはgpt-6-luna mediumでPASS、指摘なし。仕様・提示HTML・コード根拠・検査証拠の照合とNode構文検査を独立実行。ブラウザ検査/実時間遷移は主担当の証拠であり、独立再実行は未実施（session/対象SHAはspec.json）。CLI終了時の通知hookにWindowsのファイル名長エラーが出たが、レビュー本文のPASSと構文確認は完了しており、レビュー工程の失敗とは扱わない。
- 旧HTML/元checkoutは開始前hashと一致し、Claudeのworktreeには書き込んでいない。実験コード/数値設定/goldenに変更なし。HTML専用のため数値pytest/Ruffは未実施。stage/commit/pushしていない。
