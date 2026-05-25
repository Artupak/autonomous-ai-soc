"""Otonom SOC Motoru -- ana calisma modulu.

Dort modda calisir:
  scan     -- CloudTrail loglarini local disk'ten tek seferlik tarar
  daemon   -- Local disk'i surekli izler, SIGTERM ile graceful shutdown
  simulate -- Gercek kayitlarla canli anomali testi
  s3daemon -- AWS S3'ten gercek zamanli CloudTrail log tuketir (production)

Guvenlik mimarisi:
  - IP dogrulama    : ipaddress modulu ile format kontrolu (command injection onlemi)
  - Atomik yazma    : tempfile + os.replace ile crash-safe ban DB
  - TTL mekanizmasi : 24 saatlik otomatik ban kaldirma
  - Audit trail     : RotatingFileHandler ile tum mudahaleler kayit altinda
  - Cift katmanli yanit:
      1. OS firewall (iptables/netsh) -- lokal koruma
      2. AWS-native response          -- gercek kaynak izolasyonu
      Her ikisi bagimsiz calisir; biri basarisiz olsa digeri devreye girer.

s3daemon env degiskenleri:
  SOC_S3_BUCKET     : zorunlu -- orn. my-cloudtrail-bucket
  SOC_S3_PREFIX     : opsiyonel -- orn. AWSLogs/123456789012/CloudTrail/
  SOC_S3_REGION     : opsiyonel -- orn. eu-west-1
  SOC_POLL_INTERVAL : opsiyonel -- varsayilan 30 saniye
"""

import copy
import glob
import gzip
import ipaddress
import json
import logging
import logging.handlers
import os
import signal
import subprocess
import tempfile
import time
import argparse
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import torch

from aws_response import execute_aws_response
from config import (
    AUDIT_LOG_FILE,
    BAN_TTL_HOURS,
    BANNED_DB_FILE,
    ENCODER_PATH,
    MODEL_PATH,
    SCAN_INTERVAL,
    SCALER_PATH,
    SEQUENCE_LENGTH,
    TARGET_LOG_DIR,
    THRESHOLD_PATH,
    WHITELIST,
    setup_logging,
)
from feature_extractor import FeatureExtractor
from model import AnomalyLSTMAutoencoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IP dogrulama
# ---------------------------------------------------------------------------


def is_valid_ip(ip: str) -> bool:
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Ban veritabani
# ---------------------------------------------------------------------------


def load_and_clean_db() -> dict:
    if not os.path.exists(BANNED_DB_FILE):
        return {}
    try:
        with open(BANNED_DB_FILE, "r") as f:
            db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    now = datetime.now()
    cleaned = {
        ip: data
        for ip, data in db.items()
        if now - datetime.fromisoformat(data["timestamp"]) < timedelta(hours=BAN_TTL_HOURS)
    }
    save_db(cleaned)
    return cleaned


def save_db(db: dict) -> None:
    dir_name = os.path.dirname(BANNED_DB_FILE) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(db, f, indent=4)
        os.replace(tmp, BANNED_DB_FILE)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

_audit_logger: logging.Logger | None = None


def _audit_log(message: str) -> None:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = logging.getLogger("soc.audit")
        _audit_logger.setLevel(logging.INFO)
        _audit_logger.propagate = False
        handler = logging.handlers.RotatingFileHandler(
            AUDIT_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        _audit_logger.addHandler(handler)
    _audit_logger.info(message)


# ---------------------------------------------------------------------------
# OS firewall (lokal koruma katmani)
# ---------------------------------------------------------------------------


def execute_real_firewall_ban(ip: str) -> bool:
    if not is_valid_ip(ip):
        _audit_log(f"FW_REJECT | Gecersiz IP: {ip}")
        return False
    try:
        if os.name == "nt":
            subprocess.run(
                [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name=SOC_BLOCK_{ip}", "dir=in", "action=block", f"remoteip={ip}",
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            cmd = "ip6tables" if ":" in ip else "iptables"
            subprocess.run(
                [cmd, "-A", "INPUT", "-s", ip, "-j", "DROP"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        _audit_log(f"FW_FAIL | IP: {ip} | {e}")
        return False


# ---------------------------------------------------------------------------
# Savunma orkestratoru -- cift katmanli yanit
# ---------------------------------------------------------------------------


def execute_defense(
    ip: str,
    event: str,
    loss: float,
    threshold: float,
    db: dict,
    record: dict | None = None,
) -> None:
    """Tehdit tespitinde iki katmanli yanit uygular.

    Katman 1 -- OS firewall:
        Analiz makinesinin network stack'ini korur.
        Root/admin yetkisi gerektirir; yetersizse sadece loglanir.

    Katman 2 -- AWS-native response:
        Tehdidin tipine gore CloudTrail reaktivasyon, IAM deny-all,
        Security Group izolasyonu gibi cloud-native action'lar alir.
        record parametresi zorunlu; yoksa bu katman atlanir.

    Her iki katman bagimsiz calisir. Biri basarisiz olsa digeri
    devreye girer. En az biri basarili olursa IP ban DB'ye eklenir.
    """
    if ip in WHITELIST or ip in db:
        return
    if not is_valid_ip(ip):
        logger.warning("Gecersiz IP atlanıyor: %s", ip)
        return

    # Katman 1: OS firewall
    fw_ok = execute_real_firewall_ban(ip)

    # Katman 2: AWS-native response
    aws_ok = False
    if record is not None:
        region = os.getenv("AWS_DEFAULT_REGION")
        response_result = execute_aws_response(ip, event, loss, record, region=region)
        aws_ok = response_result.success
    else:
        logger.debug("Record bilgisi eksik, AWS response katmani atlanıyor.")

    if fw_ok or aws_ok:
        db[ip] = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "loss": loss,
            "fw": fw_ok,
            "aws": aws_ok,
        }
        save_db(db)
        _audit_log(
            f"BAN | IP: {ip} | Event: {event} | Loss: {loss:.4f} | FW: {fw_ok} | AWS: {aws_ok}"
        )
        logger.warning(
            "[BAN] IP: %s | Olay: %s | Skor: %.4f | FW: %s | AWS: %s",
            ip, event, loss, fw_ok, aws_ok,
        )
    else:
        _audit_log(f"RESPONSE_FAIL | IP: {ip} | Event: {event} | Loss: {loss:.4f}")
        logger.error(
            "[RESPONSE BASARISIZ] IP: %s | Olay: %s -- hic bir yanit eylemi calismadi.", ip, event
        )


# ---------------------------------------------------------------------------
# Analiz
# ---------------------------------------------------------------------------


def analyze_records(
    records: list[dict],
    model: AnomalyLSTMAutoencoder,
    fe: FeatureExtractor,
    threshold: float,
    device: torch.device,
    banned_db: dict,
) -> int:
    log_buf: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    meta_buf: deque[dict] = deque(maxlen=SEQUENCE_LENGTH)

    for record in records:
        event = record.get("eventName", "Unknown")
        ip = record.get("sourceIPAddress", "Unknown")

        if ip in WHITELIST or ip in banned_db:
            continue

        log_buf.append(fe.transform_record(record))
        meta_buf.append({"ip": ip, "event": event, "record": record})

        if len(log_buf) >= SEQUENCE_LENGTH:
            tensor = torch.FloatTensor(np.array(log_buf)).unsqueeze(0).to(device)
            with torch.no_grad():
                loss = torch.mean(torch.pow(tensor - model(tensor), 2)).item()
            if loss > threshold:
                t = meta_buf[-1]
                execute_defense(
                    t["ip"], t["event"], loss, threshold, banned_db,
                    record=t["record"],
                )

    return len(records)


# ---------------------------------------------------------------------------
# Mod: scan
# ---------------------------------------------------------------------------


def run_scan(
    model: AnomalyLSTMAutoencoder, fe: FeatureExtractor, threshold: float, device: torch.device
) -> None:
    logger.info("=== SCAN MODU | Dizin: %s ===", TARGET_LOG_DIR)
    banned_db = load_and_clean_db()
    files = glob.glob(os.path.join(TARGET_LOG_DIR, "**", "*.json"), recursive=True)

    if not files:
        logger.warning("JSON dosyasi bulunamadi: %s", TARGET_LOG_DIR)
        return

    total, errors = 0, 0
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                records = json.load(f).get("Records", [])
            total += analyze_records(records, model, fe, threshold, device, banned_db)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logger.error("Dosya okunamadi: %s -- %s", fpath, e)
            errors += 1

    logger.info("Tarama bitti. Islenen: %d | Hata: %d | Banli: %d", total, errors, len(banned_db))


# ---------------------------------------------------------------------------
# Mod: daemon (local disk)
# ---------------------------------------------------------------------------


def run_daemon(
    model: AnomalyLSTMAutoencoder, fe: FeatureExtractor, threshold: float, device: torch.device
) -> None:
    logger.info("=== DAEMON MODU | Esik: %.6f | Aralik: %ds ===", threshold, SCAN_INTERVAL)
    running = True

    def _stop(signum: int, frame: object) -> None:
        nonlocal running
        logger.info("Sinyal alindi (%d), kapatiliyor...", signum)
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    banned_db = load_and_clean_db()
    seen: set[str] = set()

    while running:
        for fpath in glob.glob(os.path.join(TARGET_LOG_DIR, "**", "*.json"), recursive=True):
            if fpath in seen:
                continue
            seen.add(fpath)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    records = json.load(f).get("Records", [])
                if records:
                    analyze_records(records, model, fe, threshold, device, banned_db)
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                pass

        for _ in range(SCAN_INTERVAL):
            if not running:
                break
            time.sleep(1)

    logger.info("Daemon kapatildi.")


# ---------------------------------------------------------------------------
# Mod: simulate
# ---------------------------------------------------------------------------


def run_simulate(
    model: AnomalyLSTMAutoencoder, fe: FeatureExtractor, threshold: float, device: torch.device
) -> None:
    logger.info("=== SIMULASYON MODU ===")
    files = glob.glob(os.path.join(TARGET_LOG_DIR, "**", "*.json"), recursive=True)
    if not files:
        logger.error("Log dosyasi bulunamadi: %s", TARGET_LOG_DIR)
        return

    sample: list[dict] = []
    for jf in files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                sample.extend(json.load(f).get("Records", []))
            if len(sample) >= SEQUENCE_LENGTH + 3:
                break
        except (FileNotFoundError, json.JSONDecodeError):
            continue

    if len(sample) < SEQUENCE_LENGTH:
        logger.error("Yeterli kayit yok (gereken: %d, mevcut: %d)", SEQUENCE_LENGTH, len(sample))
        return

    buf: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    logger.info("-- Normal trafik --")
    for rec in sample[: SEQUENCE_LENGTH + 1]:
        buf.append(fe.transform_record(rec))
        if len(buf) < SEQUENCE_LENGTH:
            continue
        seq = torch.FloatTensor(np.array([list(buf)])).to(device)
        with torch.no_grad():
            mse = torch.mean(torch.pow(seq - model(seq), 2)).item()
        status = "ANOMALI" if mse > threshold else "NORMAL"
        logger.info(
            "  [%s] %s / %s | MSE: %.6f",
            status, rec.get("sourceIPAddress", "?"), rec.get("eventName", "?"), mse,
        )

    logger.info("-- Saldiri trafigi --")
    attacks = [
        {"eventName": "DeleteTrail",     "sourceIPAddress": "185.10.20.30",  "errorCode": "AccessDenied",      "userIdentity": {"type": "Root"},         "awsRegion": "ap-southeast-1"},
        {"eventName": "StopLogging",     "sourceIPAddress": "91.234.56.78",  "errorCode": "AccessDenied",      "userIdentity": {"type": "IAMUser"},      "awsRegion": "eu-west-3"},
        {"eventName": "CreateUser",      "sourceIPAddress": "45.33.32.156",  "errorCode": "UnauthorizedAccess", "userIdentity": {"type": "AssumedRole"},  "awsRegion": "us-west-2"},
        {"eventName": "DisableKey",      "sourceIPAddress": "2001:db8::1",   "errorCode": "AccessDenied",      "userIdentity": {"type": "Root"},         "awsRegion": "sa-east-1"},
        {"eventName": "PutBucketPolicy", "sourceIPAddress": "103.75.200.10", "errorCode": "AccessDenied",      "userIdentity": {"type": "FederatedUser"}, "awsRegion": "af-south-1"},
    ]

    for scenario in attacks:
        fake = copy.deepcopy(sample[0])
        fake.update(scenario)
        buf.append(fe.transform_record(fake))
        seq = torch.FloatTensor(np.array([list(buf)])).to(device)
        with torch.no_grad():
            mse = torch.mean(torch.pow(seq - model(seq), 2)).item()
        tag = "TEHDIT" if mse > threshold else "TEMIZ"
        logger.info(
            "  [%s] %s / %s | MSE: %.4f",
            tag, scenario["sourceIPAddress"], scenario["eventName"], mse,
        )

    logger.info("Simulasyon bitti.")


# ---------------------------------------------------------------------------
# Mod: s3daemon
# ---------------------------------------------------------------------------

_S3_SEEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_seen_keys.json")
_S3_MAX_KEYS = 50_000


def _s3_load_seen() -> set[str]:
    try:
        with open(_S3_SEEN_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def _s3_save_seen(keys: set[str]) -> None:
    key_list = sorted(keys)[-_S3_MAX_KEYS:]
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(_S3_SEEN_FILE) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(key_list, f)
        os.replace(tmp, _S3_SEEN_FILE)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _s3_fetch(s3_client: object, bucket: str, key: str) -> list[dict]:
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)  # type: ignore[attr-defined]
        body = resp["Body"].read()
        if key.endswith(".gz"):
            body = gzip.decompress(body)
        return json.loads(body.decode("utf-8")).get("Records", [])
    except Exception as e:
        logger.error("S3 key okunamadi: %s -- %s", key, e)
        return []


def run_s3daemon(
    model: AnomalyLSTMAutoencoder, fe: FeatureExtractor, threshold: float, device: torch.device
) -> None:
    try:
        import boto3
    except ImportError:
        logger.critical("boto3 yuklu degil. 'pip install boto3' calistirin.")
        return

    bucket = os.getenv("SOC_S3_BUCKET", "").strip()
    if not bucket:
        logger.critical("SOC_S3_BUCKET env degiskeni eksik.")
        return

    prefix = os.getenv("SOC_S3_PREFIX", "")
    region = os.getenv("SOC_S3_REGION") or None
    poll_interval = int(os.getenv("SOC_POLL_INTERVAL", "30"))

    logger.info("=== S3 DAEMON | s3://%s/%s | Poll: %ds ===", bucket, prefix, poll_interval)

    s3 = boto3.Session(region_name=region).client("s3")

    try:
        s3.head_bucket(Bucket=bucket)
    except Exception as e:
        logger.critical("S3 bucket erisim hatasi: %s -- %s", bucket, e)
        return

    banned_db = load_and_clean_db()
    seen = _s3_load_seen()
    logger.info("Onceki oturumdan %d key yuklendi.", len(seen))

    running = True
    dirty = False

    def _stop(signum: int, frame: object) -> None:
        nonlocal running
        logger.info("Sinyal alindi (%d), kapatiliyor...", signum)
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    total = 0

    while running:
        try:
            paginator = s3.get_paginator("list_objects_v2")
            new_keys = [
                obj["Key"]
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
                for obj in page.get("Contents", [])
                if obj["Key"].endswith((".json", ".json.gz")) and obj["Key"] not in seen
            ]

            for key in new_keys:
                if not running:
                    break
                records = _s3_fetch(s3, bucket, key)
                if records:
                    count = analyze_records(records, model, fe, threshold, device, banned_db)
                    total += count
                    logger.info(
                        "key=%s | %d kayit | toplam=%d | banli=%d",
                        key.split("/")[-1], count, total, len(banned_db),
                    )
                seen.add(key)
                dirty = True

            if dirty:
                _s3_save_seen(seen)
                dirty = False

        except Exception as e:
            logger.error("S3 poll hatasi: %s", e)

        for _ in range(poll_interval):
            if not running:
                break
            time.sleep(1)

    _s3_save_seen(seen)
    logger.info("S3 daemon kapatildi. Toplam: %d kayit islendi.", total)


# ---------------------------------------------------------------------------
# Giris noktasi
# ---------------------------------------------------------------------------


def _load_model(device: torch.device, fe: FeatureExtractor) -> AnomalyLSTMAutoencoder:
    model = AnomalyLSTMAutoencoder(fe.scaler.n_features_in_).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Otonom SOC Motoru")
    parser.add_argument(
        "--mode",
        choices=["scan", "daemon", "simulate", "s3daemon"],
        default="daemon",
        help="Calisma modu (varsayilan: daemon)",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("SOC Motoru baslatiliyor -- Mod: %s", args.mode.upper())

    for path, label in [
        (MODEL_PATH, "Model"),
        (SCALER_PATH, "Scaler"),
        (THRESHOLD_PATH, "Threshold"),
        (ENCODER_PATH, "Encoder"),
    ]:
        if not os.path.exists(path):
            logger.error("Eksik dosya: %s (%s) -- once trainer.py calistirin.", path, label)
            return

    try:
        with open(THRESHOLD_PATH, "r") as f:
            threshold = json.load(f)["threshold"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error("Threshold okunamadi: %s", e)
        return

    fe = FeatureExtractor.load(SCALER_PATH, ENCODER_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device, fe)

    logger.info(
        "Model hazir | Cihaz: %s | Girdi: %d | Esik: %.6f",
        device, fe.scaler.n_features_in_, threshold,
    )

    modes = {
        "scan": run_scan,
        "daemon": run_daemon,
        "simulate": run_simulate,
        "s3daemon": run_s3daemon,
    }
    modes[args.mode](model, fe, threshold, device)


if __name__ == "__main__":
    main()
