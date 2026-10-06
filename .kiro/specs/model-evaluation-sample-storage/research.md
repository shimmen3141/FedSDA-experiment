# 根拠と境界

- 旧clients/base.py:42–43のstored_dataはサーバ交差評価用、train_data_storeは学習用の別owner。
- _store_evaluation_data:150–160。負IDは何もせず戻る。非負IDでは先に空列を登録し、min(入力件数,EVAL_STORE_SAMPLE_SIZE)が0なら抽出なし。それ以外はrandom.sample（全件でも実行）→extend→超過時は末尾STORED_DATA_LIMIT件を残す。
- confirm_model_registration:448–449。元があればpop→先代入。先既存は位置保持/上書き、新先・同IDは末尾。列連結・容量再抽出なし。
- apply_server_mapping:697–704。一回対応、元挿入順の連結。空列も保持。先初出順に、超過した列だけrandom.sample(capacity)、超過しなければ順序と乱数状態を保持。循環/連鎖を再帰追跡しない。
- _cross_evaluation_data:527以降の評価標本>5/現行学習標本>10 fallback/EVAL_MAX_SAMPLES制限は選択方針の別責務。今回は含めない。
- 新srcに評価標本ownerは未実装。Tensorの意味を持つrecordは評価用として独立宣言し、学習用ObservedTrainingSampleを名称だけ流用しない。payloadは借用し検査しない。
- 正常な正capacityと非負追加抽出件数を移植。旧capacity=0の[-0:]を無制限として受理する契約にはせず新入力で拒否する。旧正常経路の不具合は現段階で未観測。
