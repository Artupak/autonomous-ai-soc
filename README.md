<p align="center">
  <h1 align="center">artupak — AI-Powered Autonomous SOC Engine</h1>
  <p align="center">
    LSTM Autoencoder tabanli anomali tespit sistemi ile AWS CloudTrail loglarini<br>
    gercek zamanli analiz eden ve tehditleri otonom olarak engelleyen guvenlik motoru.(BETA- GELİŞTİRME AŞAMASINDA)
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/scikit--learn-1.3-f7931e?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/F1--Score-0.97-brightgreen" alt="F1-Score">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Hakkinda

**artupak**, bir Security Operations Center (SOC) analistinin yapacagi temel gorevleri yapay zeka ile otomatiklestiren bir guvenlik motorudur. Sistem su adimlari otonom olarak gerceklestirir:

1. **Izleme** — AWS CloudTrail log dosyalarini surekli veya tek seferlik tarar
2. **Analiz** — Her log kaydini LSTM Autoencoder modelinden gecirerek anomali skoru hesaplar
3. **Karar** — Skor, dinamik esik degerini asarsa tehdidi tanimlar
4. **Mudahale** — Tehdit kaynagi IP adresini isletim sistemi guvenlik duvarindan otomatik olarak banlar

> **Not:** Bu proje egitim amaciyla gelistirilmistir. Uretim ortaminda kullanmadan once kapsamli test ve yapilandirma yapilmasi onemle tavsiye edilir.

---

## Mimari

```
                    +-------------------+
                    |   trainer.py      |
                    |   (Egitim)        |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
         model.pth      scaler.pkl    encoders.pkl
              |              |              |
              +--------------+--------------+
                             |
                    +--------+----------+
                    |     main.py       |
                    |  (3 Calisma Modu) |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
          --mode          --mode         --mode
           scan           daemon        simulate
              |              |              |
      Tek seferlik    Surekli izleme   Canli ag testi
        tarama         (varsayilan)
```

### Model Mimarisi — LSTM Autoencoder

| Katman | Boyut | Aciklama |
|--------|-------|----------|
| Encoder LSTM | input_dim -> 128 | Girdi dizisini sikistirilmis temsile kodlar |
| Bottleneck | 128 -> 64 | Darbogazda ozellik ozeti cikarir |
| Expand | 64 -> 128 | Gizli temsili decoder icin genisletir |
| Decoder LSTM | 128 -> 128 | Sikistirilmis temsili yeniden yapilandirir |
| Output | 128 -> input_dim | Orijinal boyuta geri dondurur |
| Aktivasyon | LeakyReLU(0.1) | Dead ReLU sorununu onler |

**Calisma Prensibi:** Model yalnizca *normal* trafik verileriyle egitilir. Inference sirasinda gelen veriyi yeniden yapilandirmaya (reconstruct) calisir. Eger veri normalden sapiyorsa, yeniden yapilandirma hatasi (MSE loss) yukselir ve esik degerini astiginda **anomali** olarak isaretlenir.

---

## Performans Metrikleri

| Metrik | Deger |
|--------|-------|
| **F1-Score** | 0.9767 |
| **Precision** | 0.9545 |
| **Recall** | 1.0000 |
| **ROC-AUC** | 0.9964 |
| **Dinamik Esik** | 6.0569 |
| **Ozellik Sayisi** | 173 (711'den dusuk varyans filtresi sonrasi) |

```
Confusion Matrix:
                 Tahmin: Normal    Tahmin: Anomali
Gercek: Normal       2366               17
Gercek: Anomali         0              357
```

---

## Kurulum

### Gereksinimler

- Python 3.10+
- pip

### Yerel Kurulum

```bash
# Repoyu klonla
git clone https://github.com/<kullanici>/artupak-soc.git
cd artupak-soc

# Bagimliliklari kur
pip install -r requirements.txt
```

### Docker ile Kurulum

```bash
# Imaji olustur ve calistir
docker-compose up --build -d

# Loglari izle
docker logs -f artupak-soc
```

---

## Kullanim

### 1. Model Egitimi

Modeli sifirdan egitmek veya yeniden egitmek icin:

```bash
python trainer.py
```

Bu komut asagidaki dosyalari olusturur:

| Dosya | Icerik |
|-------|--------|
| `cloud_ai_V2_model.pth` | Egitilmis model agirliklari |
| `cloud_ai_V2_scaler.pkl` | StandardScaler parametreleri |
| `cloud_ai_V2_encoders.pkl` | LabelEncoder'lar ve sutun listesi |
| `cloud_ai_V2_threshold.json` | Dinamik anomali esik degeri |

### 2. Tek Seferlik Tarama (Scan)

CloudTrail log dizinindeki tum JSON dosyalarini bir kez tarar:

```bash
python main.py --mode scan
```

### 3. Surekli Izleme — Daemon (Varsayilan)

Arka planda surekli calisarak yeni log dosyalarini otomatik tespit eder:

```bash
python main.py --mode daemon
```

- Her **10 saniyede** bir dizini kontrol eder
- Sadece **yeni** dosyalari isler (daha once tarananlar atlanir)
- Tehdit tespitinde **isletim sistemi guvenlik duvarina** kural yazar
- Ban suresi: **24 saat** (TTL mekanizmasi ile otomatik temizlenir)

### 4. Canli Ag Simulasyonu

Modelin gercek zamanli ag trafiginde nasil calisacagini gosterir:

```bash
python main.py --mode simulate
```

Normal trafik paketleri gonderildikten sonra saldirgandan gelen anormal paketler simule edilir.

---

## Proje Yapisi

```
artupak-soc/
|
|-- config.py                  # Merkezi konfigurasyon ve logging ayarlari
|-- model.py                   # LSTM Autoencoder model sinifi
|-- feature_extractor.py       # Egitim/inference arasi tutarli ozellik donusumu
|-- trainer.py                 # Model egitim motoru
|-- main.py                    # Ana calisma dosyasi (scan / daemon / simulate)
|
|-- tests/                     # Test suite (pytest)
|   |-- __init__.py
|   |-- conftest.py            # Paylasimli fixture'lar (ban DB izolasyonu vb.)
|   +-- test_core.py           # IP dogrulama, ban DB, feature, model testleri
|
|-- cloud_ai_V2_model.pth      # [Uretilen] Egitilmis model agirliklari
|-- cloud_ai_V2_scaler.pkl     # [Uretilen] StandardScaler
|-- cloud_ai_V2_encoders.pkl   # [Uretilen] LabelEncoder paketi
|-- cloud_ai_V2_threshold.json # [Uretilen] Anomali esik degeri
|
|-- banned_ips_daemon.json     # Ban veritabani (TTL destekli)
|-- ai_defense_audit.log       # Guvenlik denetim gunlugu (10 MB rotasyon)
|
|-- aws_kaggle_dataset/        # Egitim veri seti
|   |-- supervise_data.csv     #   Etiketli CloudTrail verileri (~17 MB)
|   +-- unsupervise_data.csv   #   Etiketsiz CloudTrail verileri (~10 MB)
|
|-- aws_dataset-main/          # Inference icin ham CloudTrail loglari
|   +-- CloudTrail/            #   55 adet JSON log dosyasi
|
|-- .github/workflows/ci.yml   # GitHub Actions CI pipeline
|-- .pre-commit-config.yaml    # Pre-commit (black, ruff, mypy) yapilandirmasi
|-- Dockerfile                 # Docker imaj tanimi
|-- docker-compose.yml         # Docker Compose yapilandirmasi
|-- pyproject.toml             # Python paket tanimlamasi
|-- requirements.txt           # Python bagimliliklari
|-- .gitignore                 # Git disinda tutulan dosyalar
|-- LICENSE                    # MIT lisansi
+-- README.md                  # Bu dosya
```

---

## Guvenlik Mekanizmalari

| Mekanizma | Aciklama |
|-----------|----------|
| **IP Dogrulama** | Firewall kurali yazilmadan once IPv4/IPv6 format kontrolu yapilir (command injection onlemi) |
| **Whitelist** | AWS dahili servisleri ve bilinen guvenli kaynaklar otomatik olarak atlanir |
| **TTL Mekanizmasi** | Banlar 24 saat sonra otomatik olarak kaldirilir (kalici banlama onlenir) |
| **Denetim Gunlugu** | Tum tehdit tespitleri `ai_defense_audit.log`'a yazilir (10 MB rotasyon, 5 yedek) |
| **Deterministik Kodlama** | `hash()` yerine kayitli `LabelEncoder` kullanilir (tutarli sonuclar) |
| **Model Yukleme** | `torch.load(weights_only=True)` ile pickle saldirilarina karsi koruma. `joblib.load` icin dosya butunlugu kontrolu kullanicinin sorumlulugundadir |
| **Ayricalik Gereksinimleri** | Firewall islemleri icin root/admin yetkisi gereklidir |
| **Atomik Dosya Yazimi** | Ban veritabani crash-safe atomik yazma ile guncellenir (`tempfile` + `os.replace`) |
| **Dusuk Varyans Filtresi** | Bilgi tasimayan sutunlar egitim sirasinda otomatik cikarilir (inference dogrulugu icin kritik) |
| **Reproducibility** | Tum seed'ler sabitlenmis (`torch`, `numpy`, `random`), ayni sonuclar garanti |
| **Kod Kalitesi** | `pre-commit` (ruff, black, trailing-whitespace) ve `mypy` ile tam tip guvenligi saglanmistir |

---

## Simulasyon Modu Hakkinda Not

`python main.py --mode simulate` komutu, ayni gercek AWS CloudTrail JSON sablonunu kullanarak hem normal hem de saldiri verisi uretir. Saldiri senaryolarinda yalnizca (IP, Event Name, vb.) kucuk birkac ozellik degistirilirken, modelin analiz ettigi 173 ozelligin %95'i (region, user agent yapilari) **ayni kalmaktadir**. Bu sebeple, simulasyon modunda **saldiri skorlari ile normal skorlar arasindaki fark bilerek dusuk tutulmustur**. 

Bu durum kodla veya model mimarisiyle ilgili bir hata degildir; modelin genel yapiyi ezberleyip sadece IP degisti diye anomali caldirmadiginin bir gostergesidir. Farkli yapisal formatlarda gelen gercek kayitlarda model gercek anomali esigini asabilmektedir.

---

## Veri Akisi

```
CloudTrail JSON Kaydi
        |
        v
FeatureExtractor.transform_record()
  - JSON anahtarlarini CSV sutunlarina esle
  - Kayitli LabelEncoder ile kategorik degerleri kodla
  - StandardScaler ile olceklendir
        |
        v
173 boyutlu ozellik vektoru
        |
        v
Kayan Pencere Tamponu (5 adim)
        |
        v
LSTM Autoencoder (Yeniden Yapilandirma)
        |
        v
MSE Loss Hesapla
        |
        +---> Loss <= threshold ---> TEMIZ (Islem yok)
        |
        +---> Loss > threshold  ---> TEHDIT TESPITI
                                    |
                                    v
                              execute_defense()
                                |       |       |
                                v       v       v
                            Firewall  Ban DB  Audit Log
```

---

## Teknik Detaylar

### Egitim Pipeline'i

1. `supervise_data.csv` (~17 MB, 720 sutun) okunur
2. Gereksiz sutunlar dusurulur (`EventId`, `EventTime`, vb.)
3. Kategorik sutunlar `LabelEncoder` ile sayisallastirilir
4. Varyansi sifir veya ihmal edilebilir duzeyde olan sutunlar cikarilir (711 -> 173)
5. Kalan sutunlar `StandardScaler` ile normalize edilir
6. 5'lik kayan pencere (sliding window) ile zaman dizileri olusturulur
7. Veri once %80/%20 oraninda egitim/test olarak bolunur, ardindan egitim setinin %15'i dogrulama icin ayrilir (sonuc: ~%68 egitim, ~%12 dogrulama, %20 test)
8. Test setine %15 oraninda sentetik anomali eklenir
9. 120 epoch boyunca MSE loss ile egitilir
10. Dinamik esik: `val_mse_mean + 3 * val_mse_std`

### Egitim Hiperparametreleri

| Parametre | Deger |
|-----------|-------|
| Batch Size | 256 |
| Epoch | 120 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Gradient Clipping | max_norm=1.0 |
| Min Variance Filtresi | 1e-6 (altindaki sutunlar cikarilir) |
| Sequence Length | 5 |

---

## Gerekli Bagimliliklar

| Paket | Surum | Kullanim |
|-------|-------|----------|
| torch | 2.1.0 | LSTM Autoencoder modeli |
| scikit-learn | 1.3.1 | StandardScaler, LabelEncoder, metrikler |
| pandas | 2.1.1 | CSV veri islemleri |
| numpy | 1.26.0 | Sayisal hesaplamalar |
| joblib | 1.3.2 | Model/scaler serializasyonu |
| pytest | 9.0.3 | Test suite |

---

## Testler

Test suite'i calistirmak icin:

```bash
python -m pytest tests/ -v
```

### Kod Kalitesi (Pre-commit)
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
mypy .
```

Mevcut testler:

| Test Sinifi | Kapsam |
|-------------|--------|
| `TestIPValidation` | IPv4 dogrulama, command injection onlemi |
| `TestBanDatabase` | TTL suresi, atomik yazma, bozuk dosya yonetimi |
| `TestFeatureExtractor` | Vektor boyutu, determinizm, bilinmeyen etiket, NaN kontrolu |
| `TestModelConsistency` | Deterministik cikti, gurultu-temiz MSE karsilastirmasi |
| `TestAnalyzeRecords` | Kayit sayisi, whitelist atlama, tehdit tespiti, tekrar banlama onlemi |
| `TestExecuteDefense` | Whitelist korumasi, gecersiz IP reddi, firewall mock dogrulamasi |

---

## Lisans

Bu proje MIT lisansi altinda lisanslanmistir.
