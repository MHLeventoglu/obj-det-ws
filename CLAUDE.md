# CLAUDE.md

Bu, object detection modellerini eğitmek/test etmek/deploy etmek için kullanılan
bir workspace'tir. Birden fazla detection mimarisi (D-FINE, RT-DETR, RT-DETRv4,
RF-DETR, YOLOv11) aynı tutarlı dizin kalıbıyla yönetilir.

Detaylı dizin anlatımı ve dataset format varsayımları için: `WORKSPACE_STRUCTURE.md`.

## Dizin Yapısı

```text
obj-det-ws/
├── models/          # Kaynak kodu workspace içinde tutulan modeller (git submodule benzeri klonlar)
│   ├── D-FINE/
│   ├── RT-DETR/
│   └── RT-DETRv4/
│
├── configs/         # Her mimari için YAML config dosyaları
│   ├── d-fine/
│   ├── rf-detr/
│   ├── rt-detr/
│   ├── rt-detrv4/
│   └── yolov11/
│
├── scripts/         # Her model için train/test/infer/deploy scriptleri
│   ├── d-fine/      #   train.py · test.py · infer.py · deploy.py
│   ├── rf-detr/
│   ├── rt-detr/
│   ├── rt-detrv4/
│   └── yolov11/
│
├── tools/           # Dataset hazırlama yardımcıları
│   ├── yolo_to_coco.py
│   ├── slice_dataset.py
│   └── patch_dfine_obj365_ids.py
│
├── notebooks/       # Colab eğitim notebookları + egitim-log.md
│
├── WORKSPACE_STRUCTURE.md
├── models-to-train.md
└── CLAUDE.md
```

## Temel Kalıp

Her model aynı üçlü kalıbı izler:

- `configs/<model>/` — deney/koşum başına bir YAML config.
- `scripts/<model>/` — `train.py`, `test.py`, `infer.py`, `deploy.py`.
- `models/<model>/` — **yalnızca** kaynak kodu workspace içinde tutulan modeller için
  (D-FINE, RT-DETR, RT-DETRv4). RF-DETR ve YOLOv11 pip kütüphanesi üzerinden
  kullanılır, `models/` altında kaynak kodu yoktur.

Yeni model eklerken aynı kalıbı tekrarla: `configs/<model>/`, `scripts/<model>/`
ve gerekiyorsa `models/<model>/`.

Config dosyası adlandırması: `<arch>_<dataset>.{yml,yaml}` (örn.
`dfine_hgnetv2_m_datasetv1.yml`, `yolo11m_datasetv1.yaml`). Dataset tanımı ile
model eğitim configi ayrı dosyalardır (D-FINE/RT-DETR ailesinde
`*_detection.yml` dataset tanımıdır).

## Çalıştırma

Tüm scriptler workspace kökünden çalıştırılır ve varsayılan olarak kendi model
configini kullanır:

```bash
python scripts/<model>/train.py                       # varsayılan config
python scripts/<model>/train.py --config configs/<model>/<custom>.yml
python scripts/yolov11/train.py --dry-run             # komutu çalıştırmadan göster
```

- D-FINE / RT-DETR / RT-DETRv4 scriptleri ilgili `models/<model>/` kaynak ağacını
  `torchrun`/subprocess ile sarmalar. D-FINE train scripti çalıştırmadan önce
  `tools/patch_dfine_obj365_ids.py` ile obj365 kategori eşlemesini patch'ler.
- YOLOv11 / RF-DETR scriptleri ilgili pip kütüphanesini doğrudan çağırır.

## datasetv1 Sınıf Sırası

```text
0: arac   1: insan   2: uap   3: uai
```

D-FINE ve RT-DETRv4 COCO `category_id` değerlerini doğrudan kullanır; bu yüzden
`category_id` `0..3` aralığında olmalı. Aileye göre beklenen dataset formatları
(`datasets/...` yerleşimi) için `WORKSPACE_STRUCTURE.md`'ye bak.

## Notlar

- Yazışma ve doküman dili Türkçe.
- Eğitim koşumları `notebooks/egitim-log.md` içinde loglanır; Colab notebookları
  W&B takibini varsayılan açar (`WANDB_API_KEY` Colab Secrets'tan okunur).
