# Object Detection Workspace Structure

Bu workspace simdilik uc ana bolumden olusur:

- `models/`: Kaynak kodu workspace icinde tutulacak modeller burada tutulur.
- `configs/`: Her mimari icin YAML config dosyalari burada tutulur.
- `scripts/`: Her model icin train, test, inference ve deployment scriptleri burada tutulur.

## Dizin Yapisi

```text
obj-det-ws/
├── models/
│   ├── D-FINE/
│   └── RT-DETRv4/
│
├── configs/
│   ├── d-fine/
│   │   ├── datasetv1_detection.yml
│   │   ├── dfine_hgnetv2_m_datasetv1.yml
│   │   └── README.md
│   │
│   ├── rf-detr/
│   │   ├── rfdetr_medium_datasetv1.yaml
│   │   └── README.md
│   │
│   ├── rt-detrv4/
│   │   ├── datasetv1_detection.yml
│   │   ├── rtv4_hgnetv2_m_datasetv1.yml
│   │   └── README.md
│   │
│   └── yolov11/
│       ├── datasetv1.yaml
│       ├── yolo11m_datasetv1.yaml
│       ├── datasetv1_sliced_1080.yaml
│       ├── yolo11m_datasetv1_sliced_1080.yaml
│       └── README.md
│
├── scripts/
│   ├── d-fine/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── infer.py
│   │   └── deploy.py
│   │
│   ├── rf-detr/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── infer.py
│   │   └── deploy.py
│   │
│   ├── rt-detrv4/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── infer.py
│   │   └── deploy.py
│   │
│   └── yolov11/
│       ├── train.py
│       ├── test.py
│       ├── infer.py
│       └── deploy.py
│
├── tools/
│   ├── yolo_to_coco.py
│   └── slice_dataset.py
│
├── notebooks/
│   ├── dfine_m_1080_sliced_colab.ipynb
│   └── yolo11m_1080_sliced_colab.ipynb
│
├── WORKSPACE_STRUCTURE.md
└── models-to-train.md
```

## Kullanim Mantigi

Yeni bir model eklendiginde ayni kalip takip edilir:

```text
configs/<model-name>/
scripts/<model-name>/
```

Eger modelin kaynak kodu workspace icinde tutulacaksa ek olarak:

```text
models/<model-source-name>/
```

RF-DETR ve YOLOv11 gibi Python kutuphanesi uzerinden kullanilacak modeller icin `models/` altinda kaynak kod dizini gerekmez.

Her model config klasorunde farkli deneyler veya kosular icin YAML dosyalari tutulur:

```text
configs/d-fine/dfine_hgnetv2_m_datasetv1.yml
configs/rt-detrv4/rtv4_hgnetv2_m_datasetv1.yml
configs/rf-detr/rfdetr_medium_datasetv1.yaml
configs/yolov11/yolo11m_datasetv1.yaml
```

## datasetv1 Varsayimi

datasetv1 sinif sirasi:

```text
0: arac
1: insan
2: uap
3: uai
```

D-FINE ve RT-DETRv4 COCO annotation dosyalarinda `category_id` degerlerini dogrudan kullanir. Bu nedenle `category_id` degerleri `0..3` araliginda olmali.

Model ailelerine gore beklenen dataset formatlari:

```text
D-FINE / RT-DETRv4:
datasets/datasetv1/train/train.json
datasets/datasetv1/val/val.json

D-FINE 1080x1080 sliced:
datasets/datasetv1_sliced_1080/train/train.json
datasets/datasetv1_sliced_1080/val/val.json

RF-DETR:
datasets/datasetv1/rf-detr/train/_annotations.coco.json
datasets/datasetv1/rf-detr/valid/_annotations.coco.json
datasets/datasetv1/rf-detr/test/_annotations.coco.json

YOLOv11:
datasets/datasetv1/yolo/images/train
datasets/datasetv1/yolo/images/val
datasets/datasetv1/yolo/labels/train
datasets/datasetv1/yolo/labels/val

YOLOv11 1080x1080 sliced:
datasets/datasetv1_sliced_1080/yolo/images/train
datasets/datasetv1_sliced_1080/yolo/images/val
datasets/datasetv1_sliced_1080/yolo/labels/train
datasets/datasetv1_sliced_1080/yolo/labels/val
```

Her model script klasorunde su dosyalar bulunur:

- `train.py`: Egitim akisi.
- `test.py`: Test veya evaluation akisi.
- `infer.py`: Inference akisi.
- `deploy.py`: Deployment veya export akisi.

Scriptler varsayilan olarak kendi model config dosyasini kullanir:

```bash
python scripts/yolov11/train.py
```

Farkli bir config dosyasi vermek icin:

```bash
python scripts/yolov11/train.py --config configs/yolov11/custom-dataset.yaml
```
