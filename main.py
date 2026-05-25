"""artupak Otonom SOC Motoru -- ana calisma modulu.

Uc modda calisir:
  scan     -- CloudTrail loglarini tek seferlik tarar
  daemon   -- Surekli izleme dongusu, SIGTERM ile graceful shutdown (varsayilan)
  simulate -- Gercek kayitlarla ag simulasyonu

Guvenlik mimarisi:
  - IP dogrulama: ipaddress modulu ile format kontrolu (command injection onlemi)
  - Atomik yazma: tempfile + os.replace ile crash-safe ban DB guncelleme
  - TTL mekanizmasi: Kalici ban yerine 24 saatlik otomatik temizleme
  - Audit trail: Tum mudahaleler RotatingFileHandler ile kayit altinda
  - Firewall butunlugu: Kural yazilamassa IP ban DB'ye eklenmez
"""

import os
import json
import glob
import copy
import ipaddress
import time
import signal
import tempfile
import argparse
import subprocess
import logging
import logging.handlers
from datetime import datetime, timedelta
from collections import deque

import torch
import numpy as np

from config import (
    MODEL_PATH,
    SCALER_PATH,
    THRESHOLD_PATH,
    ENCODER_PATH,
    SEQUENCE_LENGTH,
    SCAN_INTERVAL,
    BAN_TTL_HOURS,
    BANNED_DB_FILE,
    AUDIT_LOG_FILE,
    TARGET_LOG_DIR,
    WHITELIST,
    setup_logging,
)
from model import AnomalyLSTMAutoencoder
from feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# ===================================================================
#  IP DOGRULAMA
# ===================================================================


def is_valid_ip(ip: str) -> bool:
    """IPv4 ve IPv6 format dogrulamasi. Command injection onlemi icin zorunlu."""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


# ===================================================================
#  BAN VERITABANI (TTL DESTEKLI)
# ===================================================================


def load_and_clean_db() -> dict:
    """Ban DB'yi yukle, suresi dolmus kayitlari temizle."""
    if not os.path.exists(BANNED_DB_FILE):
        return {}

    try:
        with open(BANNED_DB_FILE, "r") as f:
            db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    now = datetime.now()
    cleaned = {}
    for ip, data in db.items():
        banned_at = datetime.fromisoformat(data["timestamp"])
        if now - banned_at < timedelta(hours=BAN_TTL_HOURS):
            cleaned[ip] = data

    save_db(cleaned)
    return cleaned


def save_db(db: dict) -> None:
    """Ban veritabanini atomik olarak diske yaz.

    Gecici dosyaya yaz, sonra os.replace ile atomik takas yap.
    Crash sirasinda bile dosya bozulmaz.
    """
    dir_name = os.path.dirname(BANNED_DB_FILE) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(db, f, indent=4)
        os.replace(tmp_path, BANNED_DB_FILE)
    except Exception:
        # Gecici dosya kaldiysa temizle
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ===================================================================
#  FIREWALL MOTORU
# ===================================================================


def execute_real_firewall_ban(ip_address: str) -> bool:
    """OS seviyesinde firewall kurali yaz. Basarisizlik audit log'a duser."""
    if not is_valid_ip(ip_address):
        _audit_log(f"FW_REJECT | Gecersiz IP formati: {ip_address}")
        return False

    try:
        if os.name == "nt":
            subprocess.run(
                [
                    "netsh",
                    "advfirewall",
                    "firewall",
                    "add",
                    "rule",
                    f"name=artupak_BLOCK_{ip_address}",
                    "dir=in",
                    "action=block",
                    f"remoteip={ip_address}",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            # IPv6 adresleri icin ip6tables kullan
            fw_cmd = "ip6tables" if ":" in ip_address else "iptables"
            subprocess.run(
                [fw_cmd, "-A", "INPUT", "-s", ip_address, "-j", "DROP"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        _audit_log(f"FW_FAIL | IP: {ip_address} | Hata: {e}")
        return False


# Audit logger -- RotatingFileHandler ile log rotasyonu
# 10 MB'a ulasinca otomatik rotate, son 5 dosya tutulur
_audit_logger: logging.Logger | None = None


def _get_audit_logger() -> logging.Logger:
    """Audit logger'i lazy olarak olustur."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = logging.getLogger("artupak.audit")
        _audit_logger.setLevel(logging.INFO)
        _audit_logger.propagate = False
        handler = logging.handlers.RotatingFileHandler(
            AUDIT_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        _audit_logger.addHandler(handler)
    return _audit_logger


def _audit_log(message: str) -> None:
    """Denetim gunlugune kayit ekle. 10 MB'da otomatik rotasyon."""
    _get_audit_logger().info(message)


def execute_defense(
    ip_address: str, event_name: str, loss_score: float, threshold: float, db: dict
) -> None:
    """Otonom savunma orkestratoru: whitelist/ban kontrol, firewall, log."""
    if ip_address in WHITELIST or ip_address in db:
        return

    if not is_valid_ip(ip_address):
        logger.warning("Gecersiz IP, atlanıyor: %s", ip_address)
        return

    fw_ok = execute_real_firewall_ban(ip_address)

    if fw_ok:
        # Firewall basarili, ban DB'ye kaydet
        db[ip_address] = {
            "timestamp": datetime.now().isoformat(),
            "event": event_name,
            "loss": loss_score,
        }
        save_db(db)

        _audit_log(
            f"BAN | IP: {ip_address} | FW: SUCCESS | "
            f"Event: {event_name} | Loss: {loss_score:.4f}"
        )

        logger.warning(
            "[OTONOM MUDAHALE] IP: %s | Olay: %s | Skor: %.4f > %.4f | FW: SUCCESS",
            ip_address,
            event_name,
            loss_score,
            threshold,
        )
    else:
        # Firewall basarisiz, IP ban DB'ye eklenmedi
        _audit_log(
            f"FW_FAIL_NO_BAN | IP: {ip_address} | " f"Event: {event_name} | Loss: {loss_score:.4f}"
        )

        logger.warning(
            "[FW BASARISIZ] IP: %s banlanaMADI, firewall kurali yazilamadi. "
            "Olay: %s | Skor: %.4f > %.4f",
            ip_address,
            event_name,
            loss_score,
            threshold,
        )


# ===================================================================
#  ANALIZ FONKSIYONU
# ===================================================================


def analyze_records(
    records: list[dict],
    model: AnomalyLSTMAutoencoder,
    feature_extractor: FeatureExtractor,
    threshold: float,
    device: torch.device,
    banned_db: dict,
) -> int:
    """CloudTrail kayitlarini modelden gecirerek anomali analizi yap."""
    # deque(maxlen) otomatik olarak eski elemanlari atar, pop(0) gerekmez
    log_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    meta_buffer: deque[dict] = deque(maxlen=SEQUENCE_LENGTH)
    count = 0

    for record in records:
        event = record.get("eventName", "Unknown")
        ip = record.get("sourceIPAddress", "Unknown")

        if ip == "Unknown":
            logger.debug("sourceIPAddress alani eksik, kayit yine de analiz edilecek")

        if ip in WHITELIST or ip in banned_db:
            count += 1
            continue

        vec = feature_extractor.transform_record(record)
        log_buffer.append(vec)
        meta_buffer.append({"ip": ip, "event": event})
        count += 1

        if len(log_buffer) >= SEQUENCE_LENGTH:
            tensor = torch.FloatTensor(np.array(log_buffer)).unsqueeze(0).to(device)
            with torch.no_grad():
                recon = model(tensor)
                loss = torch.mean(torch.pow(tensor - recon, 2)).item()

            if loss > threshold:
                trigger = meta_buffer[-1]
                execute_defense(trigger["ip"], trigger["event"], loss, threshold, banned_db)

    return count


# ===================================================================
#  MOD: SCAN
# ===================================================================


def run_scan(
    model: AnomalyLSTMAutoencoder,
    fe: FeatureExtractor,
    threshold: float,
    device: torch.device,
) -> None:
    """CloudTrail log dizinini tek seferlik tara."""
    logger.info("=== SCAN MODU ===")
    logger.info("Hedef dizin: %s", TARGET_LOG_DIR)

    banned_db = load_and_clean_db()
    json_files = glob.glob(os.path.join(TARGET_LOG_DIR, "**", "*.json"), recursive=True)

    if not json_files:
        logger.warning("Hedef dizinde JSON dosyasi bulunamadi: %s", TARGET_LOG_DIR)
        return

    total = 0
    errors = 0

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                records = json.load(f).get("Records", [])
            if records:
                total += analyze_records(records, model, fe, threshold, device, banned_db)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logger.error("Dosya okunamadi: %s -- %s", fpath, e)
            errors += 1

    logger.info(
        "Tarama tamamlandi. Islenen: %d, Hata: %d, Banli: %d",
        total,
        errors,
        len(banned_db),
    )


# ===================================================================
#  MOD: DAEMON
# ===================================================================


def run_daemon(
    model: AnomalyLSTMAutoencoder,
    fe: FeatureExtractor,
    threshold: float,
    device: torch.device,
) -> None:
    """Surekli izleme dongusu. Yeni dosyalari otomatik isle."""
    logger.info("=== DAEMON MODU ===")
    logger.info("Cihaz: %s | Esik: %.6f | Aralik: %ds", device, threshold, SCAN_INTERVAL)

    # Graceful shutdown icin sinyal yonetimi
    running = True

    def _handle_signal(signum, frame):
        nonlocal running
        logger.info("Kapatma sinyali alindi (signal=%d), duzgun kapatiliyor...", signum)
        running = False

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    banned_db = load_and_clean_db()
    seen: set[str] = set()

    while running:
        try:
            files = glob.glob(os.path.join(TARGET_LOG_DIR, "**", "*.json"), recursive=True)
            new_files = [f for f in files if f not in seen]

            for fpath in new_files:
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        records = json.load(f).get("Records", [])
                    if records:
                        analyze_records(records, model, fe, threshold, device, banned_db)
                except (FileNotFoundError, json.JSONDecodeError, ValueError):
                    pass
                seen.add(fpath)

            # Kapatma sinyali sleep sirasinda da kontrol edilsin
            for _ in range(SCAN_INTERVAL):
                if not running:
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Daemon durduruldu.")
            break

    logger.info("Daemon duzgun kapatildi.")


# ===================================================================
#  MOD: SIMULATE
# ===================================================================


def run_simulate(
    model: AnomalyLSTMAutoencoder,
    fe: FeatureExtractor,
    threshold: float,
    device: torch.device,
) -> None:
    """Gercek CloudTrail kayitlariyla ag simulasyonu.

    Normal trafik icin gercek log kayitlari kullanilir.
    Saldiri trafigi icin kayitlar degistirilerek (event adi, IP, errorCode)
    modelin anomali tespiti test edilir.
    """
    logger.info("=== SIMULASYON MODU ===")

    # Gercek kayitlari yukle
    json_files = glob.glob(os.path.join(TARGET_LOG_DIR, "**", "*.json"), recursive=True)
    if not json_files:
        logger.error("Simulasyon icin log dosyasi bulunamadi: %s", TARGET_LOG_DIR)
        return

    sample_records: list[dict] = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                recs = json.load(f).get("Records", [])
                sample_records.extend(recs)
            if len(sample_records) >= SEQUENCE_LENGTH + 3:
                break
        except (FileNotFoundError, json.JSONDecodeError):
            continue

    if len(sample_records) < SEQUENCE_LENGTH:
        logger.error(
            "Yeterli kayit yok (en az %d gerekli, mevcut: %d)",
            SEQUENCE_LENGTH,
            len(sample_records),
        )
        return

    traffic: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

    # --- Normal trafik: gercek CloudTrail kayitlari ---
    n_normal = min(SEQUENCE_LENGTH + 1, len(sample_records))
    logger.info("Normal trafik gonderiliyor (%d kayit)...", n_normal)

    for i in range(n_normal):
        rec = sample_records[i]
        vec = fe.transform_record(rec)
        traffic.append(vec)

        ip = rec.get("sourceIPAddress", "N/A")
        event = rec.get("eventName", "N/A")

        if len(traffic) < SEQUENCE_LENGTH:
            logger.info(
                "  Paket %d (%s / %s) -> Tampon dolmadi (%d/%d)",
                i + 1,
                ip,
                event,
                len(traffic),
                SEQUENCE_LENGTH,
            )
            continue

        seq = torch.FloatTensor(np.array([list(traffic)])).to(device)
        with torch.no_grad():
            recon = model(seq)
            mse = torch.mean(torch.pow(seq - recon, 2)).item()

        status = "ANOMALI" if mse > threshold else "NORMAL"
        logger.info("  Paket %d (%s / %s) -> %s (MSE: %.6f)", i + 1, ip, event, status, mse)

    # --- Saldiri trafigi: degistirilmis kayitlar ---
    logger.info("")
    logger.info("Saldiri trafigi gonderiliyor...")

    # Farkli saldiri senaryolari -- her biri mumkun oldugunca cok alani degistirir
    attack_scenarios = [
        {
            "eventName": "DeleteTrail",
            "sourceIPAddress": "185.10.20.30",
            "errorCode": "AccessDenied",
            "errorMessage": "User is not authorized",
            "userIdentity": {"type": "Root", "invokedBy": "unknown"},
            "awsRegion": "ap-southeast-1",
            "readOnly": "false",
        },
        {
            "eventName": "StopLogging",
            "sourceIPAddress": "91.234.56.78",
            "errorCode": "AccessDenied",
            "userIdentity": {"type": "IAMUser", "invokedBy": "manual"},
            "awsRegion": "eu-west-3",
            "readOnly": "false",
        },
        {
            "eventName": "CreateUser",
            "sourceIPAddress": "45.33.32.156",
            "errorCode": "UnauthorizedAccess",
            "userIdentity": {"type": "AssumedRole", "invokedBy": "unknown"},
            "awsRegion": "us-west-2",
            "readOnly": "false",
        },
        {
            "eventName": "DisableKey",
            "sourceIPAddress": "2001:db8::dead:beef",
            "errorCode": "AccessDenied",
            "userIdentity": {"type": "Root"},
            "awsRegion": "sa-east-1",
            "readOnly": "false",
        },
        {
            "eventName": "PutBucketPolicy",
            "sourceIPAddress": "103.75.200.10",
            "errorCode": "AccessDenied",
            "errorMessage": "User is not authorized",
            "userIdentity": {"type": "FederatedUser", "invokedBy": "unknown"},
            "awsRegion": "af-south-1",
            "readOnly": "false",
        },
    ]

    for i, scenario in enumerate(attack_scenarios):
        # Gercek kaydin derin kopyasini al, saldiri alanlariyla degistir
        fake = copy.deepcopy(sample_records[0])
        for key, val in scenario.items():
            fake[key] = val

        atk_ip = scenario["sourceIPAddress"]
        atk_event = scenario["eventName"]

        vec = fe.transform_record(fake)
        traffic.append(vec)

        seq = torch.FloatTensor(np.array([list(traffic)])).to(device)
        with torch.no_grad():
            recon = model(seq)
            mse = torch.mean(torch.pow(seq - recon, 2)).item()

        if mse > threshold:
            logger.warning(
                "  [TEHDIT] %s / %s -> MSE: %.4f (esik: %.4f)",
                atk_ip,
                atk_event,
                mse,
                threshold,
            )
        else:
            logger.info("  [TEMIZ] %s / %s -> MSE: %.6f", atk_ip, atk_event, mse)

    logger.info("Simulasyon tamamlandi.")


# ===================================================================
#  ANA GIRIS NOKTASI
# ===================================================================


def main() -> None:
    """Komut satiri argumanlari, model yukleme, mod secimi."""
    parser = argparse.ArgumentParser(description="artupak Otonom SOC Motoru")
    parser.add_argument(
        "--mode",
        choices=["scan", "daemon", "simulate"],
        default="daemon",
        help="Calisma modu (varsayilan: daemon)",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("artupak SOC Motoru baslatiliyor -- Mod: %s", args.mode.upper())

    # Kritik dosya kontrolu
    for path, label in [
        (MODEL_PATH, "Model"),
        (SCALER_PATH, "Scaler"),
        (THRESHOLD_PATH, "Threshold"),
        (ENCODER_PATH, "Encoder"),
    ]:
        if not os.path.exists(path):
            logger.error(
                "Kritik dosya eksik: %s (%s). Once trainer.py'yi calistirin.",
                path,
                label,
            )
            return

    # Esik yukle
    try:
        with open(THRESHOLD_PATH, "r") as f:
            threshold = json.load(f)["threshold"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error("Esik degeri okunamadi: %s", e)
        return

    # Feature extractor yukle
    fe = FeatureExtractor.load(SCALER_PATH, ENCODER_PATH)
    input_dim = fe.scaler.n_features_in_

    # Model yukle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyLSTMAutoencoder(input_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    logger.info("Model yuklendi (%s). Girdi: %d, Esik: %.6f", device, input_dim, threshold)

    modes = {"scan": run_scan, "daemon": run_daemon, "simulate": run_simulate}
    modes[args.mode](model, fe, threshold, device)


if __name__ == "__main__":
    main()
