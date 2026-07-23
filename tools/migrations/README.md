# 結果名称の移行

`migrate_results.py`は、過去の`results/`に含まれる手法名・データセット名だけを
現在表記へ変換する保守ツールです。パラメータ名と数値は変更しません。

最初の実行では検証済みステージングだけを作成します。

```bash
python -m tools.migrations.migrate_results
```

内容を確認した後、同じステージングを有効化します。旧`results/`は日時付きの
バックアップとして残ります。

```bash
python -m tools.migrations.migrate_results --apply
```

画像内の凡例は安全に書き換えられないため、PNGは移行せずバックアップ側に残します。
