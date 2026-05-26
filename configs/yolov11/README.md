# YOLOv11 Training

Bu klasor datasetv1 icin YOLO11m egitim configlerini tutar.

## Configler

- `datasetv1.yaml`: Ultralytics YOLO dataset tanimi.
- `yolo11m_datasetv1.yaml`: YOLO11m egitim configi.
- `datasetv1_sliced_1080.yaml`: 1080x1080 crop edilmis datasetv1 YOLO dataset tanimi.
- `yolo11m_datasetv1_sliced_1080.yaml`: 1080x1080 crop dataset ile YOLO11m egitim configi.

## Dataset

Beklenen dataset yerlesimi:

```text
datasets/datasetv1/yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Sinif sirasi:

```text
0: arac
1: insan
2: uap
3: uai
```

YOLO label dosyalari su formatta olmalidir:

```text
class_id x_center y_center width height
```

Koordinatlar normalize edilmis `0..1` araliginda olmalidir.

1080x1080 crop edilmis dataset icin beklenen yerlesim:

```text
datasets/datasetv1_sliced_1080/yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Colab uzerinden tek akista hazirlik ve egitim icin:

```text
notebooks/yolo11m_1080_sliced_colab.ipynb
```

Notebook varsayilanlari tek 40GB A100 icin ayarlanmistir:

```text
imgsz: 640
batch: 64
workers: 8
```

Notebook W&B takibini varsayilan olarak acar. Colab Secrets icinde
`WANDB_API_KEY` tanimliysa otomatik kullanilir; degilse notebook login
hucrelerinde interaktif `wandb.login()` calisir. YOLO notebooku Ultralytics
W&B entegrasyonu icin `yolo settings wandb=True` komutunu da calistirir.

## Train

Komutu once dry-run ile kontrol et:

```bash
python scripts/yolov11/train.py --dry-run
```

Varsayilan egitim:

```bash
python scripts/yolov11/train.py
```

Device secerek egitim:

```bash
python scripts/yolov11/train.py --device 0
```

Epoch, image size ve batch override:

```bash
python scripts/yolov11/train.py --epochs 100 --imgsz 640 --batch 16 --device 0
```

Worker sayisi override:

```bash
python scripts/yolov11/train.py --workers 2
```

Farkli model veya data YAML ile egitim:

```bash
python scripts/yolov11/train.py --model yolo11m.pt --data configs/yolov11/datasetv1.yaml
```

Not: YOLOv11 Ultralytics Python kutuphanesi kullanilir; bu workspace icinde YOLO kaynak kodu tutulmaz.
