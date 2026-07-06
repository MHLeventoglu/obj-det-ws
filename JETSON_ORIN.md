# D-FINE: Jetson Orin Kurulum ve Test Kılavuzu

Bu dokuman, egitilmis bir D-FINE checkpoint'ini (ozellikle tek sinifli N model,
`configs/d-fine/dfine_hgnetv2_n_plane_44k.yml`) Jetson Orin uzerinde iki farkli
yolla test etmek icin gerekli adimlari sirayla anlatir:

- **TensorRT'siz test**: PyTorch checkpoint'i ile `scripts/d-fine/test.py`
  uzerinden tam COCO degerlendirmesi (precision/recall/mAP).
- **TensorRT'li test**: Checkpoint'i once ONNX'e, sonra Jetson uzerinde
  TensorRT engine'e cevirip inference/latency testi.

> **Onemli not:** D-FINE repo'sunda TensorRT engine uzerinden COCO mAP
> hesaplayan hazir bir script yok. TRT tarafinda yalnizca tekil goruntu/video
> inference (`tools/inference/trt_inf.py`) ve gecikme benchmarki
> (`tools/benchmark/trt_benchmark.py`) mevcut. Dogruluk (mAP) karsilastirmasi
> icin referans hep PyTorch tarafindaki `scripts/d-fine/test.py`'dir.

## 0. Jetson Orin Sistem Hazirligi

1. JetPack surumunu dogrula (Orin icin JetPack 5.1.x → L4T 35.x, CUDA 11.4,
   cuDNN 8.6, TensorRT 8.5, veya JetPack 6.x → L4T 36.x, CUDA 12.2,
   TensorRT 8.6/10.x):
   ```bash
   cat /etc/nv_tegra_release
   sudo apt show nvidia-jetpack
   ```
2. CUDA ve TensorRT'nin sistemde hazir geldigini dogrula (JetPack ile birlikte
   kurulu gelirler, ayrica kurulum gerekmez):
   ```bash
   nvcc --version
   python3 -c "import tensorrt; print(tensorrt.__version__)"
   trtexec --help | head -5
   ```
3. Gerekli sistem paketleri:
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv git \
       libopenblas-dev libjpeg-dev zlib1g-dev cmake
   ```
4. Egitim/export sirasinda bellek yetmemesi ihtimaline karsi swap alanini
   buyutmek faydali olabilir (ozellikle 8GB Orin Nano gibi kartlarda):
   ```bash
   sudo systemctl disable nvzramconfig
   sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
   sudo mkswap /swapfile && sudo swapon /swapfile
   ```

## 1. Python Ortami ve Bagimliliklar

1. TensorRT'nin Python binding'i yalnizca sistem Python'unda gelir; venv'i
   `--system-site-packages` ile olustur:
   ```bash
   python3 -m venv --system-site-packages ~/venvs/dfine
   source ~/venvs/dfine/bin/activate
   ```
2. PyTorch/torchvision: PyPI'daki genel wheel'ler Jetson'in aarch64+CUDA
   kurulumuyla calismaz. NVIDIA'nin JetPack surumune uygun Jetson PyTorch
   wheel'ini (NVIDIA Jetson AI Lab / forums.developer.nvidia.com uzerinden
   JetPack surumune gore) kur, ardindan dogrula:
   ```bash
   python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
3. D-FINE'in geri kalan bagimliliklarini kur (torch/torchvision zaten yukarida
   Jetson'a ozel kuruldugu icin `requirements.txt` icindeki o iki satiri
   atlayarak kur):
   ```bash
   pip install faster-coco-eval PyYAML tensorboard scipy calflops transformers loguru
   ```
4. TensorRT engine build/inference/benchmark icin ek paketler:
   ```bash
   pip install onnx onnxsim          # ONNX export'u Jetson'da yapacaksan
   pip install onnxruntime           # tools/inference/onnx_inf.py icin (opsiyonel)
   pip install pycuda                # yalnizca tools/benchmark/trt_benchmark.py icin
   ```
   `tools/inference/trt_inf.py` video islerken `cv2` kullanir; JetPack
   imajlarinda genelde `python3-opencv` hazir gelir (`python3 -c "import cv2"`
   ile dogrula, yoksa `sudo apt install python3-opencv`).

## 2. Workspace, Checkpoint ve Dataset'i Jetson'a Tasima

```bash
# Jetson uzerinde
git clone <repo-url> obj-det-ws
cd obj-det-ws

# Egitim yapilan makineden checkpoint + val dataset'i tasi
rsync -avP <host>:output/dfine_hgnetv2_n_plane_44k/best_stg2.pth \
    output/dfine_hgnetv2_n_plane_44k/best_stg2.pth
rsync -avP <host>:datasets/plane_44k/val/ datasets/plane_44k/val/
```

## 3. TensorRT'siz Test (PyTorch, tam COCO degerlendirme)

```bash
cd obj-det-ws
python scripts/d-fine/test.py \
  --config configs/d-fine/dfine_hgnetv2_n_plane_44k.yml \
  --resume output/dfine_hgnetv2_n_plane_44k/best_stg2.pth \
  -images datasets/plane_44k/val/images \
  -ann datasets/plane_44k/val/val.json
```

Once komutu calistirmadan gormek icin `--dry-run` ekle. Cikti: D-FINE
Validator metrikleri (precision/recall/f1/iou), COCOeval mAP50:95/mAP50/mAP75,
AR@1/10/100 ve tek sinif (`plane`) icin satir bazli mAP.

## 4. TensorRT'li Test

### 4a. ONNX Export + TensorRT Engine Build

TensorRT engine'ler GPU mimarisine ve TensorRT surumune ozeldir; bu yuzden
`trtexec` engine build adimi **mutlaka Jetson'in kendisinde** calistirilmali
(baska bir makinede build edilen `.engine` Jetson'da calismaz). ONNX export
adimi ise GPU gerektirmez, ister host makinede ister Jetson'da yapilabilir.

Uctan uca, dogrudan Jetson uzerinde (checkpoint'ten ONNX'e, ardindan
`trtexec` ile FP16 engine'e):

```bash
python scripts/d-fine/deploy.py \
  --config configs/d-fine/dfine_hgnetv2_n_plane_44k.yml \
  --resume output/dfine_hgnetv2_n_plane_44k/best_stg2.pth
```

Once `--dry-run` ile komutlari gorebilirsin. FP32 engine icin `--no-fp16`
ekle (Orin'in Tensor Core'larindan yararlanmak icin FP16 varsayilan olarak
onerilir).

Eger ONNX export'u host makinede yaptiysan, yalnizca `.onnx` dosyasini
Jetson'a tasi ve orada dogrudan engine build'ine gec:

```bash
python scripts/d-fine/deploy.py --onnx best_stg2.onnx --output best_stg2.engine
```

### 4b. TensorRT Engine ile Inference / Benchmark

Tekil goruntu/video ile gorsel dogrulama (PyTorch ciktisiyla kutulari gozle
karsilastirmak icin):

```bash
cd models/D-FINE
python tools/inference/trt_inf.py \
  --trt ../../output/dfine_hgnetv2_n_plane_44k/best_stg2.engine \
  --input ornek.jpg
```

Cikti `trt_result.jpg` (veya video icin `trt_result.mp4`) olarak kaydedilir.

Gecikme (latency) benchmarki:

```bash
cd models/D-FINE/tools/benchmark
pip install -r requirements.txt   # pycuda dahil
python trt_benchmark.py \
  --infer_dir ../../../datasets/plane_44k/val/images \
  --engine_dir ../../../output/dfine_hgnetv2_n_plane_44k
```

## 5. Ozet Akis

1. JetPack + CUDA + TensorRT'nin hazir geldigini dogrula.
2. venv olustur, Jetson'a ozel torch/torchvision + kalan pip paketlerini kur.
3. Workspace'i, checkpoint'i ve val dataset'ini Jetson'a tasi.
4. TensorRT'siz: `scripts/d-fine/test.py` ile tam COCO mAP degerlendirmesi.
5. TensorRT'li: `scripts/d-fine/deploy.py` ile ONNX export + Jetson'da
   `trtexec` engine build.
6. `tools/inference/trt_inf.py` ile gorsel dogrulama, `tools/benchmark/trt_benchmark.py`
   ile gecikme olcumu.

Not: TRT engine ile PyTorch checkpoint'i arasinda dogruluk (mAP) regresyonunu
otomatik karsilastiran bir script şu an yok; ihtiyac olursa
`scripts/d-fine/test.py`'a `.engine` dosyasi kabul eden bir mod eklenebilir.
