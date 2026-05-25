"""Merkezi yapilandirma modulu.

Tum dosya yollari, egitim sabitleri, daemon parametreleri ve guvenlik
politikalari bu dosyada tanimlaniyor. Proje genelinde dagitik sabitler
yerine tek kaynak ilkesi (single source of truth) uygulanir.
"""

import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
#  Model artefakt yollari
#  Egitim sonrasi uretilen dosyalar. Inference oncesi bu dosyalarin
#  var oldugu kontrol edilir (main.py giris noktasi).
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "cloud_ai_V2_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "cloud_ai_V2_scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "cloud_ai_V2_threshold.json")
ENCODER_PATH = os.path.join(BASE_DIR, "cloud_ai_V2_encoders.pkl")

# ---------------------------------------------------------------------------
#  Egitim sabitleri
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(BASE_DIR, "aws_kaggle_dataset", "supervise_data.csv")

# CloudTrail CSV'sindeki bilgi tasimayan veya kimlik iceren sutunlar.
# Bunlar modelin ogrenmesini bozar ya da overfitting'e yol acar.
COLUMNS_TO_DROP = [
    "EventId",
    "EventTime",
    "EventSource",
    "Resources",
    "AccessKeyId",
    "Username",
    "Unnamed: 717",
    "Unnamed: 718",
]
TARGET_COL = "label"
NORMAL_CLASS = 0

# Dusuk varyansli sutunlari elemek icin alt sinir.
# Bu degerin altindaki sutunlar scaler'da sifira bolme yaparak
# inference sirasinda 10^20 seviyesinde MSE skorlarina yol acar.
MIN_VARIANCE = 1e-6

# Reproducibility -- tum random kaynaklari (torch, numpy, random)
# bu degerle sabitlenerek egitimden egiteme ayni sonuc garanti edilir.
SEED = 42

# Kayan pencere uzunlugu. Autoencoder'a beslenen zaman dizisinin
# adim sayisi. Kisa tutularak CloudTrail'in olay bazli yapisina uyum saglanir.
SEQUENCE_LENGTH = 5

# ---------------------------------------------------------------------------
#  Daemon & runtime sabitleri
# ---------------------------------------------------------------------------

# Daemon modunda ardisik taramalar arasi bekleme suresi (saniye).
SCAN_INTERVAL = 10

# Bir IP ban edildikten sonra otomatik kaldirilma suresi (saat).
# Kalici ban yerine TTL mekanizmasi kullanilir, boylece gecici
# yanlis pozitiflerin etkisi sinirlanir.
BAN_TTL_HOURS = 24

BANNED_DB_FILE = os.path.join(BASE_DIR, "banned_ips_daemon.json")
AUDIT_LOG_FILE = os.path.join(BASE_DIR, "ai_defense_audit.log")
TARGET_LOG_DIR = os.path.join(BASE_DIR, "aws_dataset-main")

# ---------------------------------------------------------------------------
#  Guvenlik politikasi -- IP whitelist
#  Bu listedeki kaynaklar analiz edilmeden atlanir. AWS'nin dahili servis
#  IP'leri (cloudtrail, ec2, vb.) false positive oranini dusurur.
#  NOT: "Unknown" bilincsizce eklenmemeli -- sourceIPAddress alani eksik
#  olan kayitlar da analiz edilmelidir (guvenlik politikasi geregi).
# ---------------------------------------------------------------------------
WHITELIST: set[str] = {
    "AWS Internal",
    "cloudtrail.amazonaws.com",
    "ec2.amazonaws.com",
    "inspector2.amazonaws.com",
    "secretsmanager.amazonaws.com",
    "rds.amazonaws.com",
    "127.0.0.1",
}


def setup_logging(level: int = logging.INFO) -> None:
    """Merkezi log yapilandirmasi.

    Tum moduller icin tek seferlik cagrilir. logging.basicConfig
    ikinci cagrimda no-op oldugu icin tekrar cagirilsa da sorun cikmaz.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
