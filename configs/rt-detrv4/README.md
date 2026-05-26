# RT-DETRv4 Training

Bu klasor datasetv1 icin RT-DETRv4 HGNetv2 M egitim configlerini tutar.

## Configler

- `datasetv1_detection.yml`: datasetv1 COCO dataset tanimi.
- `rtv4_hgnetv2_m_datasetv1.yml`: RT-DETRv4 HGNetv2 M egitim configi.

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

## Ek Gereksinim

RT-DETRv4 configi DINOv3 teacher ayarlarini icerir. Egitimden once repo icinde su yollar hazir olmalidir:

```text
models/RT-DETRv4/dinov3/
models/RT-DETRv4/pretrain/dinov3_vitb16_pretrain_lvd1689m.pth
```

## Train

Komutu once dry-run ile kontrol et:

```bash
python scripts/rt-detrv4/train.py --devices 0 --dry-run
```

Tek GPU egitim:

```bash
python scripts/rt-detrv4/train.py --devices 0
```

Coklu GPU egitim:

```bash
python scripts/rt-detrv4/train.py --devices 0,1,2,3 --nproc-per-node 4
```

Farkli config ile egitim:

```bash
python scripts/rt-detrv4/train.py --config configs/rt-detrv4/rtv4_hgnetv2_m_datasetv1.yml --devices 0
```

Checkpoint'ten devam:

```bash
python scripts/rt-detrv4/train.py --devices 0 --resume path/to/checkpoint.pth
```

Pretrained checkpoint ile fine-tuning:

```bash
python scripts/rt-detrv4/train.py --devices 0 --tuning path/to/model.pth
```

Batch size gibi config degerlerini komuttan override etmek:

```bash
python scripts/rt-detrv4/train.py --devices 0 --update train_dataloader.total_batch_size=16
```
