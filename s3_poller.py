"""S3 CloudTrail Poller -- gercek zamanli AWS log tuketime modulu.

Neden ayri bir modul?
- main.py'daki local-disk daemon aynen calisir (backward compat).
- S3 entegrasyonu tek sorumluluklari olan bu module izole edildi.
- boto3 sadece bu dosyada import ediliyor; yuklenmemisse sadece
  s3daemon modu patlıyor, diger modlar etkilenmiyor.

Mimari:
  S3 Bucket (CloudTrail prefix)
      -> list_objects_v2 ile incremental poll (sadece yeni key'ler)
      -> get_object -> gzip/json parse
      -> analyze_records() [main.py'dan alinir]
      -> execute_defense() [main.py'dan alinir]

Kimlik dogrulama onceligi (boto3 standart chain):
  1. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env degiskenleri
  2. ~/.aws/credentials profili
  3. EC2/ECS/Lambda IAM Instance Role (production'da bu kullanilir)
  Kod hic bir credential hardcode etmez.

Gerekli IAM izinleri (Least Privilege):
  - s3:GetObject      (log okuma)
  - s3:ListBucket     (yeni dosya kesfetme)
  - s3:GetBucketLocation (region dogrulama)
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import signal
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tip alias -- circular import olmadan main.py fonksiyonlarini almak icin
# ---------------------------------------------------------------------------
AnalyzeFn = Callable[..., int]


# ---------------------------------------------------------------------------
# Gorulmus key'leri diske kaydet (process restart sonrasi tekrar isleme onler)
# ---------------------------------------------------------------------------
_SEEN_KEYS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "s3_seen_keys.json"
)
_MAX_SEEN_KEYS = 50_000  # bellek siniri; en eski key'ler atilir


def _load_seen_keys() -> set[str]:
    """Daha once islenmis S3 key'lerini diskten yukle."""
    try:
        with open(_SEEN_KEYS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def _save_seen_keys(keys: set[str]) -> None:
    """Gorulmus key'leri atomik olarak diske yaz."""
    import tempfile

    key_list = list(keys)
    # Boyut limitini as inca en eski (alphabetically) key'leri at.
    # CloudTrail key'leri timestamp icerdiginden bu FIFO'ya yakindir.
    if len(key_list) > _MAX_SEEN_KEYS:
        key_list = sorted(key_list)[-_MAX_SEEN_KEYS:]

    dir_name = os.path.dirname(_SEEN_KEYS_FILE) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(key_list, f)
        os.replace(tmp, _SEEN_KEYS_FILE)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# S3 key -> CloudTrail records
# ---------------------------------------------------------------------------

def _fetch_records_from_key(s3_client: object, bucket: str, key: str) -> list[dict]:
    """S3'ten tek bir CloudTrail log dosyasini indir ve kayitlari dondur.

    CloudTrail dosyalari gzip'li JSON olarak gelir:
      { "Records": [ {...}, {...} ] }
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)  # type: ignore[attr-defined]
        body = response["Body"].read()

        # Gzip mi duzce JSON mi?
        if key.endswith(".gz"):
            body = gzip.decompress(body)

        data = json.loads(body.decode("utf-8"))
        records = data.get("Records", [])
        logger.debug("S3 key islendi: %s (%d kayit)", key, len(records))
        return records
    except Exception as exc:
        logger.error("S3 key okunamadi: %s -- %s", key, exc)
        return []


# ---------------------------------------------------------------------------
# Ana S3 daemon dongusu
# ---------------------------------------------------------------------------

def run_s3daemon(
    analyze_records_fn: AnalyzeFn,
    model: object,
    feature_extractor: object,
    threshold: float,
    device: object,
    banned_db: dict,
) -> None:
    """S3'ten gercek zamanli CloudTrail log tuketen daemon.

    Parametreler:
        analyze_records_fn : main.py'daki analyze_records fonksiyonu
        model              : yuklu AnomalyLSTMAutoencoder
        feature_extractor  : yuklu FeatureExtractor
        threshold          : dinamik anomali esigi
        device             : torch.device
        banned_db          : mevcut ban veritabani dict'i (in-place guncellenir)

    Env degiskenleri (zorunlu):
        SOC_S3_BUCKET   : CloudTrail loglarinin bulundugu S3 bucket adi
        SOC_S3_PREFIX   : Opsiyonel prefix, orn. "AWSLogs/123456789/CloudTrail/"

    Env degiskenleri (opsiyonel):
        SOC_S3_REGION   : boto3 region override (varsayilan: AWS_DEFAULT_REGION)
        SOC_POLL_INTERVAL: Taramalar arasi bekleme suresi saniye (varsayilan: 30)
    """
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        logger.critical(
            "boto3 yuklu degil. 'pip install boto3' calistirin ve tekrar deneyin."
        )
        return

    bucket = os.getenv("SOC_S3_BUCKET", "").strip()
    if not bucket:
        logger.critical(
            "SOC_S3_BUCKET env degiskeni tanimli degil. "
            "Ornek: export SOC_S3_BUCKET=my-cloudtrail-bucket"
        )
        return

    prefix = os.getenv("SOC_S3_PREFIX", "").strip()
    region = os.getenv("SOC_S3_REGION") or None
    poll_interval = int(os.getenv("SOC_POLL_INTERVAL", "30"))

    logger.info("=== S3 DAEMON MODU ===")
    logger.info(
        "Bucket: s3://%s/%s | Region: %s | Poll: %ds",
        bucket,
        prefix,
        region or "default",
        poll_interval,
    )

    # boto3 session -- IAM role veya env credential'lari otomatik kullanir
    session = boto3.Session(region_name=region)
    s3 = session.client("s3")

    # Bucket erisim testi
    try:
        s3.head_bucket(Bucket=bucket)
        logger.info("S3 bucket erisimi dogrulandi: %s", bucket)
    except (BotoCoreError, ClientError) as exc:
        logger.critical("S3 bucket'a erisim saglanamadi: %s -- %s", bucket, exc)
        return

    seen_keys: set[str] = _load_seen_keys()
    logger.info("Onceki oturumdan %d islenmis key yuklendi.", len(seen_keys))

    # Graceful shutdown
    running = True

    def _handle_signal(signum: int, frame: object) -> None:
        nonlocal running
        logger.info("Kapatma sinyali alindi (signal=%d), duzgun kapatiliyor...", signum)
        running = False

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    total_processed = 0
    new_keys_this_session: set[str] = set()

    while running:
        try:
            # Prefix altindaki tum .json ve .json.gz key'leri listele
            paginator = s3.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

            batch_new: list[str] = []
            for page in page_iterator:
                for obj in page.get("Contents", []):
                    key: str = obj["Key"]
                    if not (key.endswith(".json") or key.endswith(".json.gz")):
                        continue
                    if key in seen_keys:
                        continue
                    batch_new.append(key)

            if not batch_new:
                logger.debug("Yeni key bulunamadi, %ds bekleniyor...", poll_interval)
            else:
                logger.info("%d yeni S3 key bulundu, isleniyor...", len(batch_new))

            for key in batch_new:
                if not running:
                    break
                records = _fetch_records_from_key(s3, bucket, key)
                if records:
                    count = analyze_records_fn(
                        records, model, feature_extractor, threshold, device, banned_db
                    )
                    total_processed += count
                    logger.info(
                        "  key=%s | %d kayit islendi | toplam=%d | banli=%d",
                        key.split("/")[-1],
                        count,
                        total_processed,
                        len(banned_db),
                    )
                seen_keys.add(key)
                new_keys_this_session.add(key)

            # Her poll sonrasi seen_keys'i diske yaz (crash-safe)
            if new_keys_this_session:
                _save_seen_keys(seen_keys)
                new_keys_this_session.clear()

        except (BotoCoreError, ClientError) as exc:
            # Gecici AWS hatalari daemon'u durdurmasin
            logger.error("S3 poll hatasi (devam ediliyor): %s", exc)
        except Exception as exc:
            logger.exception("Beklenmeyen hata: %s", exc)

        # Poll interval'i 1'er saniye parcalara bol -- sinyal hizli alinsin
        for _ in range(poll_interval):
            if not running:
                break
            time.sleep(1)

    logger.info(
        "S3 daemon kapatildi. Toplam islenen: %d kayit | %d key.",
        total_processed,
        len(seen_keys),
    )
    _save_seen_keys(seen_keys)
