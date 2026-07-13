## FedSDA シーケンス図 (mermaid)

1 ラウンド = K 時刻分の逐次処理 + 1 回のサーバ集約。K は `AGG_INTERVAL`。

```mermaid
sequenceDiagram
    autonumber
    participant D as データストリーム
    participant C as クライアント c
    participant S as サーバ

    loop ラウンド r = 1 .. R
        S->>C: モデル集合 H^[r] をブロードキャスト
        Note over C: ワーキング集合を初期化<br/>H_work ← H^[r]

        loop 各時刻 t = (r-1)K+1 .. rK（per-sample 逐次）
            D->>C: 特徴 x_c^t
            Note over C: 予測 ŷ_c^t ← H_work[m_c^t](x_c^t)
            D->>C: ラベル y_c^t
            Note over C: 損失 ℓ を評価 → ADWIN で逐次ドリフト検知<br/>Assignment: 最適モデル選択 / 新規作成<br/>（FIFO で新旧概念を分割・τ ごとにミニバッチ更新）
        end

        Note over C: 最終状態から抽出<br/>更新済 H_upd^[r] ・ 新規 H_new^[r]
        C->>S: H_upd^[r], H_new^[r] をアップロード
        Note over S: FedAvg 加重平均<br/>H_upd^[r] ← FedAvg(H_c,upd^[r])<br/>H_new^[r] ← ∪_c H_c,new^[r]
        S->>C: 評価依頼（H_upd^[r] ∪ H_new^[r] を配布）
        Note over C: 受信モデルをローカルデータで現地評価
        C->>S: 評価結果（集約統計 n, Σℓ, Σℓ² のみ）
        Note over S: クラスタリング・統合（Aggregate）<br/>評価値→距離行列を構成<br/>階層的クラスタリング・マージ<br/>→ 次ラウンド H^[r+1] を構築
    end
```

**FedSDA の要点**: ドリフト検知は **ADWIN による per-sample の統計検定**で、時刻ごとに逐次
（`process_one_step`）。集約は K 時刻ごとに 1 回。新規/更新モデルはサーバの**階層的クラスタリング**で併合される。

---

## FedDrift シーケンス図 (mermaid)

対比用（本実装の `clients/feddrift.py` / `_run_batch_timestep` 準拠）。FedSDA と異なり
**バッチベース**で、`FEDDRIFT_DETECT_BATCH` 件を溜めてから検出・通信する。検出バッチ完了時は論文の R ラウンドに倣い {配布 → ローカル学習 → 集約} を `FEDDRIFT_ROUNDS` 回（既定 1）。

```mermaid
sequenceDiagram
    autonumber
    participant D as データストリーム
    participant C as クライアント c
    participant S as サーバ

    loop 検出バッチごと（ストリーム全体を通じて繰り返し）

        S->>C: グローバルモデル集合をブロードキャスト

        loop FEDDRIFT_DETECT_BATCH 回（サンプルをバッファに蓄積）
            D->>C: データ (x_c^t, y_c^t)
            Note over C: 現行モデルで予測を記録<br/>検出バッファに蓄積
        end

        Note over C: バッチ完了 → 全保持モデルの最小損失を評価<br/>最小損失 > 前バッチ + 閾値 → 新規モデル作成（未知概念）<br/>そうでなければ最良モデルへ切替（既知概念）
        C->>S: 更新済 / 新規モデルをアップロード
        Note over S: クラスタリング付き集約（モデル併合・割当）

        loop FEDDRIFT_ROUNDS 回（論文 R・既定 1）
            S->>C: グローバルモデルを配布
            Note over C: リプレイバッファからローカル学習<br/>（FEDDRIFT_DETECT_BATCH × UPDATES_PER_SAMPLE ステップ ＝ 1バッチ分の予算）
            C->>S: モデルをアップロード
            Note over S: FedAvg 集約
        end
    end
```

**FedDrift の要点**: ドリフト検知は **検出バッチ単位の最小損失の増分**（`FEDDRIFT_DETECT_BATCH`件ごと）。通信もこのバッチ完了時のみで、`FEDDRIFT_DETECT_BATCH`（検出粒度↔通信）と
`FEDDRIFT_ROUNDS`（バッチあたり収束度↔通信）が 2 つの通信軸。各変数の詳細は[hyperparameters.md](hyperparameters.md) を参照。
