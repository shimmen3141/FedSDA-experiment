# 設計

## 構成
v4から別HTMLへ派生。現在の標本/将来標本/roundのepoch、共有は別時間の例、配布→更新用状態→配布時へコピー、条件付き確率ベクトルの定義を維持する。

## 配色・固定共有
phase toneは1#ad443b/2#a95d10/3#2b783c/4#087d89/5#28678e/6#674a96。淡背景とborderを同系統へ。stage.currentはbox-shadow:inset 0 0 0 3px var(--tone)、背景tint、見出しとradioはtone、説明文字は暗色。dotとcaption番号はtoneで濃くする。レイアウトサイズを変えない。
server-branchのradio/choices.server/names登録/条件分岐を削除。6段は固定説明と段階jumpのみ。buildは全経路に共有例を追加する。全体/先頭/キー/再生は維持する。

## FIFOの配置と図解
- FIFO: x690,y660,w650,h80、右端1340は維持。title x1015,y684、古い側710/新しい側1312のラベルはy691。6セルx800/880/960/1040/1120/1200、各58×27、y698。区間dividerはx1030。データ容量とセル数は模式図で設定値ではない。
- released-sample: rect x460,y685,w150,h42、上に「はみ出す標本」、中に標本icon。外部actorとして固定しN03/N04だけ活性、全体を含む他コマはopacity.28。
- N02: FIFO内部は右端cellだけ1、他は.22。外標本.28。
- N03: 外標本1、FIFO枠は活性だが内部cellは.22。fifo-release(690,700→610,700)を活性化して取り出し対象を示す。captionは「容量を超えた最古の標本を選ぶ」。まだpopする命令順序を再現する意味ではない。
- N04: 外標本1、内部は背景。released-assign(610,706→650,706→650,645→880,645→880,620)（見出しを横切らない右側ポート）と既存assign-storeを活性化し、帰属確定/解放を示す。captionは同じ通常処理でストアへ確定してFIFOから外すとする。
- A03: 内部cell1/区間divider、既存fifo-assignを(1015,660→1015,640→880,640→880,620)へ変えて「FIFO内の区間」のlabel。外標本は薄い。
- R11: 内部cell1、既存replay（右辺1340は同じ）に「FIFO内で再較正」。外標本は薄い。
- その他は内部cell.22、FIFO actor自体の濃淡は元のframe契約に従う。N05の学習はストアからで、外標本は濃くしない。

## ソース上の意味
PendingTrainingAssignmentBuffer.appendは追加時に切り捨てない。get_sample_indices_exceeding_capacityは選択だけ。runtime/released_pending_sample_assignment.pyは超過標本を現行ストアへ吸収し、成功後にFIFOから解放する。N03/N04はこの同一処理を対象と受け渡しに分けた説明用コマであり、コードのpop→吸収という逆の命令順を意味しない。警報枝は別途FIFO区間の取扱い、再較正もFIFO内部の再生を示す。

## 検証
v4 CDP auditをv5用に派生。共有選択を除く54経路、1428コマの期待列と操作を検査。上段とdotのtone一致を検査し、背景は同一でなくstage=tint/dot active=toneかつinset3を確認する。FIFO actor/セル/外標本の実opacityを各frameで検査。N03は外への線、N04は外から帰属/ストア、A03/R11は内側だけの線で外に繋がらないことを検査。固定bbox・接続の他箱横断・文字・狭幅拡大も再検査。代表PNGは到着/超過選択/ストア確定/学習/警報区間/再較正/全体。
