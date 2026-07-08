# D-FINE Training

Bu klasor D-FINE egitim dataset ve model configlerini tutar.

## Configler

- `datasetv1_detection.yml`: datasetv1 COCO dataset tanimi.
- `dfine_hgnetv2_m_datasetv1.yml`: D-FINE M egitim configi.
- `datasetv1_sliced_1080_detection.yml`: 1080x1080 crop edilmis datasetv1 COCO dataset tanimi.
- `dfine_hgnetv2_m_datasetv1_sliced_1080.yml`: 1080x1080 crop dataset ile D-FINE M egitim configi.
- `datasetv1_sliced_1080_2crop_detection.yml`: hazir 1080x1080 2-crop COCO dataset tanimi.
- `dfine_hgnetv2_m_datasetv1_sliced_1080_2crop.yml`: hazir 1080x1080 2-crop dataset ile D-FINE M egitim configi.
- `datasetv1_grid_740x600_6crop_detection.yml`: hazir 740x600 6-crop COCO dataset tanimi.
- `dfine_hgnetv2_m_datasetv1_grid_740x600_6crop.yml`: hazir 740x600 6-crop dataset ile D-FINE M egitim configi.
- `datasetv2_2crop_detection.yml`: hazir datasetv2 2-crop COCO dataset tanimi.
- `dfine_hgnetv2_m_datasetv2_2crop.yml`: hazir datasetv2 2-crop dataset ile D-FINE M egitim configi.
- `plane_44k_detection.yml`: tek sinifli plane COCO dataset tanimi.
- `dfine_hgnetv2_n_plane_44k.yml`: 44k plane dataset ile D-FINE N egitim configi.

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

Hazir crop ziplerinden cikarilan datasetler icin beklenen yerlesim:

```text
datasets/datasetv1_sliced_1080_2crop/
├── annotations/
│   ├── train.json
│   └── val.json
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

COCO `file_name` degerleri dataset root'una gore relative olmalidir:

```text
train/images/example.jpg
val/images/example.jpg
```

740x600 6-crop dataset ayni yapiyi `datasets/datasetv1_grid_740x600_6crop/`
altinda kullanir.

datasetv2 2-crop dataset icin beklenen yerlesim biraz farklidir (test split
yok, annotation dosya adlari `annotations_train.json` / `annotations_val.json`):

```text
datasets/datasetv2_2crop/
├── annotations/
│   ├── annotations_train.json
│   └── annotations_val.json
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

Plane dataset icin beklenen yerlesim:

```text
datasets/plane_44k/
├── annotations/
│   ├── train.json
│   └── val.json
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

Plane COCO annotation dosyalarinda tek sinif `category_id: 0` olmalidir:

```text
0: plane
```

`notebooks/dfine_n_plane_44k_colab.ipynb` annotation yoksa YOLO
label dosyalarindan bu COCO JSON dosyalarini otomatik olusturur.

Prepared zipler Drive'dan local workspace'e acilacak sekilde notebookta
`DATASET_VARIANT` ile secilir:

```text
datasetv1_sliced_1080_2crop      train: ~18000, val: ~6500
datasetv1_grid_740x600_6crop     train: ~54000, val: ~19500
datasetv2_2crop                  train: ~22140 (approximate, val count unknown)
```

Colab uzerinden tek akista hazirlik ve egitim icin:

```text
notebooks/dfine_m_1080_sliced_colab.ipynb
notebooks/dfine_n_plane_44k_colab.ipynb
```

Notebook varsayilanlari tek 40GB A100 icin ayarlanmistir:

```text
train total_batch_size: 32
val total_batch_size: 64
num_workers: 8
```

D-FINE N plane notebook varsayilanlari:

```text
pretrained: dfine_n_coco.pth
epochs: 72
train total_batch_size: 64
val total_batch_size: 128
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
