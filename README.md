# 🏀 Basketball Scoring Recognition — Full Project

End-to-end repo for training an **R(2+1)D-18** 3D CNN on the **Basketball-51** dataset.
It includes a classic ML folder layout: `data/raw`, `data/processed/train|val|test/<class>`, and training/eval tools.

## Dataset classes (8)
`2p0, 2p1, 3p0, 3p1, ft0, ft1, mp0, mp1` → (range × make=1/miss=0).

## Quickstart — Colab
1. Open GPU runtime. Install deps:
```bash
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install av decord opencv-python tqdm scikit-learn matplotlib
!apt-get -yqq install ffmpeg
```
2. Download dataset with Kaggle token:
```bash
# Upload kaggle.json when prompted
!python -m src.data_utils.kaggle_download
```
3. Split into train/val/test (70/15/15 by default):
```bash
!python -m src.data_utils.split_dataset --raw data/raw --out data/processed --train 0.7 --val 0.15 --test 0.15
```
4. Train:
```bash
!python -m src.train --data_root data/processed --epochs 10 --batch_size 4 --lr 3e-4 --amp
```
5. Evaluate on test split:
```bash
!python -m src.evaluate --data_root data/processed --weights checkpoints/best.pt
```
6. Inference on one clip:
```bash
!python -m src.infer --weights checkpoints/best.pt --video "data/processed/val/3p1/<somefile>.mp4" --topk 5
```
7. Export ONNX:
```bash
!python -m src.export_onnx --weights checkpoints/best.pt --out model.onnx
```

## Local setup
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# make sure ffmpeg is installed (brew/apt/choco)
python -m src.data_utils.kaggle_download       # downloads to data/raw
python -m src.data_utils.split_dataset --raw data/raw --out data/processed
python -m src.train --data_root data/processed --epochs 10 --amp
python -m src.evaluate --data_root data/processed --weights checkpoints/best.pt
```

## Layout
```
.
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ config.py
│  ├─ datasets.py
│  ├─ models.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ infer.py
│  ├─ export_onnx.py
│  └─ data_utils/
│     ├─ kaggle_download.py
│     └─ split_dataset.py
└─ data/
   ├─ raw/                 # Kaggle zip + extracted content
   └─ processed/
      ├─ train/<class>/    # auto-filled by split script
      ├─ val/<class>/
      └─ test/<class>/
```
