# ローカル学習要求の保留と実行回数

## 目的と範囲
研究実装の保守者が、旧clientに混在している要求の累積・実行間隔・反復回数算出を独立して確認できるようにする。
完成済みの共同学習反復は算出済み回数しか受け取れない。今回は一標本の処理に対応して外側が明示的に発行する学習要求を数え、
間隔到達と明示flushで必要な共同更新試行回数を返す。成功後の確認により保留件数を消化する。
要求件数は観測標本の総数・学習ストア件数・FIFO未帰属件数ではない。旧train_stepの呼出し一回を一要求とする。
モデル/optimizer/標本所有、実学習、自動callback、警報/ラウンド境界の呼出し位置、通信、診断counter、新全体runは含めない。
単一所有者の同期処理を対象とし、並行/再入可能な実行は保証しない。旧基準748c3aa/goldenを変更しない。

## 1. 保留・実行回数
1.1 When 学習要求を一件記録する, the Local Training Request Schedule shall 保留要求件数を一件増やし、設定した間隔未満なら実行回数0、間隔以上なら現在の全保留件数に一要求あたりの共同更新回数を掛けた実行回数を返す。
1.2 The Local Training Request Schedule shall 算出回数を共同更新の試行回数として扱い、参加モデル数・batch標本数・成功更新数を掛けない。
1.3 When 外側が明示flush用の回数を照会する, the Local Training Request Schedule shall 間隔未満でも全保留件数から算出し、保留0なら0を返す。照会だけで保留件数を変更しない。
1.4 When 一要求あたりの共同更新回数が0である, the Local Training Request Schedule shall 要求を通常どおり保留し、算出回数0を返す。正常に完了確認した保留要求は消化できる。
1.5 When 外側が現在の全保留件数と等しい正の完了要求件数を確認する, the Local Training Request Schedule shall 保留を0へ戻す。実行回数の算出だけで要求を消化しない。

## 2. 状態とエラー
2.1 If 実行間隔が1未満または一要求あたりの共同更新回数が負、あるいはどちらかがboolを含むPython組み込み整数以外である, the Local Training Request Schedule shall 初期化を拒否する。整数派生型も受理しない。
2.2 If 完了確認の件数が正のPython組み込み整数でない、または現在の保留件数と等しくない, the Local Training Request Schedule shall 保留件数を変更せず拒否する。
2.3 While 完了確認を受けていない, the Local Training Request Schedule shall 実学習が失敗した場合も保留件数を維持する。途中まで完了した更新のrollbackや自動retryを行わない。
2.4 The Local Training Request Schedule shall 初期保留件数を0とし、実行間隔と一要求あたりの回数を構築時の不変設定として保持する。

## 3. 移植と接続
3.1 The Local Training Request Schedule shall 旧train_step/flush_pending_updatesの要求列と照合でき、間隔到達・端数flush・空flush・失敗後の再試行で呼出し順・保留件数・算出反復回数を維持する。
3.2 The Local Training Request Schedule shall 旧実装・グローバル設定・モデル/optimizer・上位実行処理に依存せず単独で回数を管理する。算出回数を完成済み共同学習反復へ外側から明示接続できる。

## 主担当要求gate
PASS。全11条件のEARS・正常/異常・0/端数/失敗・同期所有・隣接境界を確認。
実学習成功と保留消化の順序は旧コードを根拠とし、新しい実行方式は追加しない。

