# 統合版の来場者カウンター

手元の精度改善版 `accurate_counter` に、[Mac版](https://github.com/Koki0724/mitafes/tree/84f080f4dbb01c9247ea6dd7bfe8492a5af5aa7f) の運用面の工夫を取り込んだ、録画向けの統合版です。

ガラス側は全画面と通過帯の2入力、姿勢と胴体による追跡、遮蔽後に現れる人物の通過救済を維持しています。Mac版は元の追跡実装と共通で、新しい精度改善を含む版ではありません。そのため、追跡をMac版の古い方式へ戻す変更はしていません。

## 取り込んだ改善

- 動画が途中で破損・終了した場合に、最後まで処理できたと誤報しない。
- 読み込み側のキューが満杯でも、停止要求で終了できる。例外時に動画を閉じる。
- 再開時に動画のパス・順序・サイズ・更新日時、モデルのSHA-256、設定、実装、依存バージョンを照合する。
- SQLiteに保存済みの位置より先の未確定ログを、再開前に切り戻す。
- 予定した観測数と実際の処理数を照合する。短い動画を長時間試験として扱わない。
- 時間帯ごとの遅延を記録し、等速入力時の遅延増加を確認できる。
- Apple Silicon/MPSとCUDAを選べる。MPSの実演算確認、Macのメモリ単位への対応を取り込む。
- 重みをチェックサムで検証して取得する。既存モデルを上書きしない。

加えて、2入力の通過を照合する処理をSQLiteの一時索引へ変更しました。映像全体の軌跡をRAMへ読み込まず、必要な時刻の軌跡を最大128件までキャッシュします。同じ22分49秒のログで、最大RSSが107.06 MiBから52.30 MiBへ減り、通過イベントと軌跡の出力は完全一致しました。索引は照合終了時に削除します。

## Linux / CUDAで実行

Python 3.11を使います。GPU・ドライバーに対応するCUDA版PyTorchを先に導入し、依存ライブラリとモデルを準備してください。検証環境はQuadro P6000 / PyTorch 2.10.0+cu126です。

```sh
git clone --branch develop-gpu https://github.com/Koki0724/mitafes.git
cd mitafes
python3.11 -m venv .venv
source .venv/bin/activate
# ここでGPUに対応するCUDA版PyTorch 2.10.0 / torchvision 0.25.0を導入
python -m pip install -r requirements.txt
python scripts/download_models.py
python doctor.py --platform cuda

# ガラス側の精度改善プリセット。最初の1分を検証
python run_accuracy.py \
  --platform cuda --gpus 0 1 \
  --models-dir ./models --source /path/to/glass.mp4 \
  --max-seconds 60 --output runs/my_glass_60s
```

`--max-seconds` を外すと末尾まで処理します。GPUが1枚なら `--gpus 0` を指定します。ガラス側のこのプリセットは、元の設置位置の2880×1620 / 25fps用です。

見下ろし側は従来の追跡・通過判定を使います。

```sh
python run.py \
  --platform cuda --devices 0 1 --batch 8 --camera overhead \
  --source /path/to/overhead.mp4 \
  --start-seconds 670 --max-seconds 60 \
  --record-tracks --output runs/my_overhead_60s
```

670秒は元の基準区間の開始位置です。別動画の先頭なら `--start-seconds 0` にします。

## Apple Siliconで実行

MPSのコード経路を実装しましたが、**この統合版をMac実機で推論した速度・精度は未検証**です。以下は実機検証用の手順です。

```sh
# arm64 Python 3.11 / macOS 14以降
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mac.txt
python scripts/download_models.py
python doctor.py --platform mps

python run_accuracy.py --platform mps \
  --models-dir ./models --source /path/to/glass.mp4 \
  --max-seconds 60 --output runs/mac_glass_60s
```

MPSではFP32・バッチ1を使います。CUDA Graphは使いません。利用できるアクセラレーターがない場合、CPU推論への自動切り替えは行いません。環境変数でMPSのCPUフォールバックを有効にした場合は、純粋なGPU推論と測定条件が異なるため、`doctor.py` の出力も保存して比較します。

モデルは `--models-dir` で指定できます。省略時はこのフォルダーの `models/` を使い、この作業環境にある既存の `../cpu_counter/models/` も利用します。Linuxで新規環境を作る場合は、GPU・ドライバーに合うPyTorchを先に導入してください。ここではQuadro P6000と `torch 2.10.0+cu126` の組み合わせを使用しています。

## 分割録画と再開

```sh
# 欠落や重複のない連続した分割録画を順に指定
python run_accuracy.py --platform cuda --gpus 0 \
  --source part01.mp4 part02.mp4 part03.mp4 \
  --max-seconds 3600 --output runs/school_day

# 同じ入力・モデル・コード・設定で処理範囲を延長
python run_accuracy.py --platform cuda --gpus 0 \
  --source part01.mp4 part02.mp4 part03.mp4 \
  --max-seconds 7200 --output runs/school_day --resume
```

各ファイルで追跡をリセットせず、同じ寸法・fpsの分割録画を連続して処理します。コードや重みを変更した場合は新しい出力先を使います。従来版のチェックポイントをこの統合版へ持ち込む方式ではありません。

2入力の状態は `full/` と `roi/` にそれぞれ保存します。`result/events.csv` が照合後の通過一覧です。`pipeline_status.json` は実行中・失敗・完了の状態と、確認するログを示します。再開時の処理時間は今回追加した部分、通過数は累計です。

## 検証結果と範囲

[VALIDATION.md](VALIDATION.md) に数値と成果物を記載しています。前回の268件中266件、99.25%という検証サンプルの結果を維持しています。今回は新しい独立した映像での精度向上を実証した作業ではありません。

31件のテストでは、従来の2画角、画面下端の人物、遮蔽後の通過、破損・読み込み失敗、ログ切り戻し、再開条件、MPSの呼び出し、索引照合を確認します。

```sh
python -m unittest test_accuracy test_streaming test_integration test_reconciliation -v
```

録画の一括処理が対象です。8時間連続運転、実際の通信断を伴うRTSP運用、Macでの長時間の発熱・推論性能は別途検証が必要です。

このブランチの推奨入口は、ガラス側が `run_accuracy.py`、見下ろし側が `run.py --camera overhead` です。ガラス側を `run.py` 単体で実行しても、2入力の照合は行いません。追加のベンチマーク・頭部モデル・ONNX・TensorRT関連スクリプトは実験用で、上記の検証済み構成には使っていません。動画の描画、ffmpegデコーダー、NVENC出力には対応するffmpegが別途必要です。

動画、モデル重み、`runs/` の実行出力はGitに含めていません。回帰テスト用の検出データは `tests/fixtures/`、小さな検証記録は `validation/gpu_20261117/` に同梱しています。
