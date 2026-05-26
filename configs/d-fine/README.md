# D-FINE Training

Bu klasor datasetv1 icin D-FINE M egitim configlerini tutar.

## Configler

- `datasetv1_detection.yml`: datasetv1 COCO dataset tanimi.
- `dfine_hgnetv2_m_datasetv1.yml`: D-FINE M egitim configi.
- `datasetv1_sliced_1080_detection.yml`: 1080x1080 crop edilmis datasetv1 COCO dataset tanimi.
- `dfine_hgnetv2_m_datasetv1_sliced_1080.yml`: 1080x1080 crop dataset ile D-FINE M egitim configi.

## Dataset

Beklenen dataset yerlesimi:

```text
datasets/datasetv1/
├── train/
│   ├── train.json
│   └── images...
└── val/
    ├── val.json
    └── images...
```

Sinif sirasi:

```text
0: arac
1: insan
2: uap
3: uai
```

`remap_mscoco_category: False` kullanildigi icin COCO annotation dosyalarinda `category_id` degerleri `0..3` araliginda olmalidir.

1080x1080 crop edilmis dataset icin beklenen yerlesim:

```text
datasets/datasetv1_sliced_1080/
├── train/
│   ├── train.json
│   └── images...
└── val/
    ├── val.json
    └── images...
```

Colab uzerinden tek akista hazirlik ve egitim icin:

```text
notebooks/dfine_m_1080_sliced_colab.ipynb
```

Notebook varsayilanlari tek 40GB A100 icin ayarlanmistir:

```text
train total_batch_size: 32
val total_batch_size: 64
num_workers: 8
```

Notebook W&B takibini varsayilan olarak acar. Colab Secrets icinde
`WANDB_API_KEY` tanimliysa otomatik kullanilir; degilse notebook login
hucrelerinde interaktif `wandb.login()` calisir.

## Train

Komutu once dry-run ile kontrol et:

```bash
python scripts/d-fine/train.py --devices 0 --dry-run
```

Tek GPU egitim:

```bash
python scripts/d-fine/train.py --devices 0
```

Coklu GPU egitim:

```bash
python scripts/d-fine/train.py --devices 0,1,2,3 --nproc-per-node 4
```

Farkli config ile egitim:

```bash
python scripts/d-fine/train.py --config configs/d-fine/dfine_hgnetv2_m_datasetv1.yml --devices 0
```

Checkpoint'ten devam:

```bash
python scripts/d-fine/train.py --devices 0 --resume path/to/checkpoint.pth
```

Pretrained checkpoint ile fine-tuning:

```bash
python scripts/d-fine/train.py --devices 0 --tuning path/to/model.pth
```

Batch size gibi config degerlerini komuttan override etmek:

```bash
python scripts/d-fine/train.py --devices 0 --update train_dataloader.total_batch_size=16
```
