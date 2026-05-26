# RF-DETR Training

Bu klasor datasetv1 icin RF-DETR Medium egitim configini tutar.

## Config

- `rfdetr_medium_datasetv1.yaml`: RF-DETR Medium egitim configi.

## Dataset

Beklenen dataset yerlesimi:

```text
datasets/datasetv1/rf-detr/
├── train/
│   ├── _annotations.coco.json
│   └── images...
├── valid/
│   ├── _annotations.coco.json
│   └── images...
└── test/
    ├── _annotations.coco.json
    └── images...
```

Sinif sirasi:

```text
0: arac
1: insan
2: uap
3: uai
```

## Train

Komutu once dry-run ile kontrol et:

```bash
python scripts/rf-detr/train.py --dry-run
```

Varsayilan egitim:

```bash
python scripts/rf-detr/train.py
```

Epoch, batch size ve learning rate override:

```bash
python scripts/rf-detr/train.py --epochs 100 --batch-size 4 --grad-accum-steps 4 --lr 0.0001
```

Farkli dataset dizini:

```bash
python scripts/rf-detr/train.py --dataset-dir datasets/datasetv1/rf-detr
```

Farkli output dizini:

```bash
python scripts/rf-detr/train.py --output-dir outputs/rf-detr/rfdetr_medium_datasetv1
```

Not: RF-DETR Python kutuphanesi kullanilir; bu workspace icinde RF-DETR kaynak kodu tutulmaz.
