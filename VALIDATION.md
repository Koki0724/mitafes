# 検証記録

## リポジトリ整備時の確認（2026-11-17）

Apple Silicon Mac（arm64 Python 3.11.15）で以下を確認した。隣接する `mitafes` の既存仮想環境を使用し、直接依存7パッケージは `requirements.txt` の固定バージョンと一致。

- `python -m unittest test_streaming test_mac -v`：13件成功。
- `python doctor.py`：`mps_smoke_test: passed`、終了コード0。MPS上での基本演算のみを確認。
- `uv pip check --python <既存仮想環境のPython>`：依存関係の不整合なし。
- `python scripts/download_models.py --check`：元の重み2種のSHA-256が一致。
- 一時ディレクトリへの新規ダウンロードでも、公式配布の重み2種のSHA-256が一致。元の重みは変更していない。

環境・確認範囲は [repository_mac_20261117.json](validation/repository_mac_20261117.json) に記録。この確認では実映像のYOLO推論、性能測定、長時間運用は行っていない。

## 初回のLinux検証

実行環境はLinux x86_64、Python 3.11.4、PyTorch 2.10.0+cpu、Ultralytics 8.3.235。**Mac実機やMPSでのYOLO推論は実行していない。処理速度の実測値はまだない。** このLinux環境を使ったことは、Mac用コードにCPU推論モードがあるという意味ではない。

## 確認済み

- `python -m unittest test_streaming test_mac -v`：13件成功。詳細は `validation/tests.txt`。
- 凍結した検出データから、ガラス側62通過、見下ろし側16通過の方向・フレーム・位置を再現。見下ろし側の傘2件の監査結果も一致。
- 追跡状態とByteTrack IDの復元、古い追跡状態の退役、奇数フレームの分割ファイル境界でのstrideを確認。
- 読み込みエラーの伝播、満杯キューの停止、MPS不可時の停止、Rosetta Pythonの拒否、MacとLinuxのRSS単位変換を確認。
- 実際の合成AVIのデコード・追跡・SQLite保存を使った再開試験が成功。推論部分は空検出を返すテスト用スタブに置換し、GPU性能を測った結果としては扱っていない。途中の未確定ログを切り戻し、フレーム0, 2, 4, 6, 8, 10, 12を一度ずつ処理。
- 軌跡評価器に既存の基準イベント・基準軌跡そのものを渡す自己整合性確認で、62/62・16/16件を照合。これはMPSの推論結果に対する評価ではない。
- `doctor.py` はこのLinux環境を拒否して終了コード1。`run.py` もMPS利用不可時は出力先を作らず停止。
- 新規Pythonファイルの構文チェックが成功。
- 固定した直接依存7パッケージすべてに、PyPI上でPython 3.11用macOS arm64 wheel（または共通wheel）があることを確認。Mac上の依存関係解決・インストールは未実行。
- 作業開始時に記録した既存の60ファイル（`realtime_counter`・`counting_poc` のコード、文書、モデル、プロファイル）のSHA-256が終了時も一致。`original_hashes.json` に記録（パスは隣接する `mitafes/` を基準とする）。`cpu_counter` への変更も行っていない。

## Macで残る確認

1. arm64 Python環境へインストールし、`doctor.py` のMPS実演算が成功すること。
2. 実映像のモデル推論がエラーなく動作し、推論器のデバイスが `mps` であること。
3. 既知の各1分で750観測を処理し、62＋16通過の軌跡が維持されること。
4. 最大速度での処理余力と、等速入力時の遅延が増え続けないこと。
5. 2画角同時実行と長時間の速度・温度・メモリ。RTSPは今回の実装の対象外。

実行手順と判定に使う項目は [README.md](README.md) を参照。
