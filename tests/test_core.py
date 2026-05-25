"""artupak SOC Motoru -- birim ve entegrasyon testleri.

Test hiyerarsisi:
  TestIPValidation     : IP format dogrulama ve injection onlemi
  TestBanDatabase      : TTL yonetimi, atomik yazma, bozuk dosya dayanikliligi
  TestFeatureExtractor : Vektor boyutu, determinizm, bilinmeyen etiket, NaN
  TestModelConsistency : Deterministik cikti, gurultu-temiz MSE farki
  TestAnalyzeRecords   : Uctan uca analiz: kayit sayisi, whitelist, tehdit, tekrar ban
  TestExecuteDefense   : Savunma orkestratoru: whitelist, gecersiz IP, firewall mock

Model dosyalarina bagli testler CI ortaminda skip olur. Bu testler
yerel ortamda egitim sonrasi calistirilmalidir.
"""

import os
import json
from unittest.mock import patch

import pytest
import numpy as np

from config import SEQUENCE_LENGTH, SCALER_PATH, ENCODER_PATH, MODEL_PATH


# ===================================================================
#  IP DOGRULAMA TESTLERI
# ===================================================================


class TestIPValidation:
    """IPv4/IPv6 format dogrulamasi ve command injection korumasini test eder."""

    def test_valid_ipv4(self):
        """Standart IPv4 adresleri kabul edilmeli."""
        from main import is_valid_ip

        assert is_valid_ip("192.168.1.1") is True
        assert is_valid_ip("10.0.0.1") is True
        assert is_valid_ip("0.0.0.0") is True
        assert is_valid_ip("255.255.255.255") is True

    def test_valid_ipv6(self):
        """IPv6 adresleri de kabul edilmeli (scoped dahil)."""
        from main import is_valid_ip

        assert is_valid_ip("::1") is True
        assert is_valid_ip("2001:db8::1") is True
        assert is_valid_ip("fe80::1%eth0") is True
        assert is_valid_ip("2600:1f18:24e6:b900:a989:3f5a:c2e9:1234") is True
        assert is_valid_ip("not:an:ipv6") is False

    def test_out_of_range(self):
        """Octet sinirini asan adresler reddedilmeli."""
        from main import is_valid_ip

        assert is_valid_ip("999.999.999.999") is False
        assert is_valid_ip("256.1.1.1") is False

    def test_malformed(self):
        """Yapisal olarak bozuk girdiler reddedilmeli."""
        from main import is_valid_ip

        assert is_valid_ip("abc.def.ghi.jkl") is False
        assert is_valid_ip("") is False
        assert is_valid_ip("192.168.1") is False
        assert is_valid_ip("192.168.1.1.1") is False

    def test_injection_attempts(self):
        """Shell injection iceren girdiler reddedilmeli.

        Bu test, firewall komutu olusturulmadan once IP dogrulamasinin
        command injection saldirilarina karsi koruma sagladigini dogrular.
        """
        from main import is_valid_ip

        assert is_valid_ip("192.168.1.1; rm -rf /") is False
        assert is_valid_ip('10.0.0.1" && echo pwned') is False
        assert is_valid_ip("$(whoami)") is False


# ===================================================================
#  BAN VERITABANI TESTLERI
# ===================================================================


class TestBanDatabase:
    """Ban DB CRUD islemleri, TTL mekanizmasi ve dayaniklilik testleri."""

    def test_load_missing_file(self, tmp_ban_db):
        """Ban DB dosyasi yoksa bos dict donmeli (ilk calistirma senaryosu)."""
        import main

        db = main.load_and_clean_db()
        assert db == {}

    def test_ttl_expiration(self, tmp_ban_db):
        """Suresi dolmus IP'ler otomatik temizlenmeli, gecerli olanlar kalmali."""
        from datetime import datetime, timedelta
        import main

        expired = (datetime.now() - timedelta(hours=BAN_TTL_HOURS + 1)).isoformat()
        valid = (datetime.now() - timedelta(minutes=5)).isoformat()

        data = {
            "1.2.3.4": {"timestamp": expired, "event": "test", "loss": 5.0},
            "5.6.7.8": {"timestamp": valid, "event": "test", "loss": 3.0},
        }
        with open(main.BANNED_DB_FILE, "w") as f:
            json.dump(data, f)

        cleaned = main.load_and_clean_db()
        assert "1.2.3.4" not in cleaned, "Suresi dolmus IP temizlenmemis"
        assert "5.6.7.8" in cleaned, "Gecerli IP yanlis temizlenmis"

    def test_atomic_save(self, tmp_ban_db):
        """save_db sonrasi dosya gecerli JSON icermeli (crash-safe yazma)."""
        import main

        payload = {
            "10.0.0.1": {
                "timestamp": "2024-01-01T00:00:00",
                "event": "x",
                "loss": 1.0,
            }
        }
        main.save_db(payload)

        with open(main.BANNED_DB_FILE, "r") as f:
            loaded = json.load(f)
        assert loaded == payload

    def test_corrupt_db_handled(self, tmp_ban_db):
        """Bozuk JSON dosyasi sessizce bos dict donmeli (dayaniklilik)."""
        import main

        with open(main.BANNED_DB_FILE, "w") as f:
            f.write("{{{invalid json")

        db = main.load_and_clean_db()
        assert db == {}


# TTL testinde kullaniliyor -- config'den import
from config import BAN_TTL_HOURS


# ===================================================================
#  FEATURE EXTRACTOR TESTLERI
# ===================================================================


class TestFeatureExtractor:
    """Ozellik cikarimi: vektor boyutu, determinizm, kenar durumlari."""

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        """Egitilmis scaler/encoder yoksa testi atla (CI ortami)."""
        if not (os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
            pytest.skip("Egitilmis model dosyalari bulunamadi")

    def _load(self):
        from feature_extractor import FeatureExtractor

        return FeatureExtractor.load(SCALER_PATH, ENCODER_PATH)

    def test_output_shape(self):
        """Bos kayit bile dogru boyutta vektor donmeli (scaler.n_features_in_)."""
        ext = self._load()
        vec = ext.transform_record({})
        assert vec.shape == (ext.scaler.n_features_in_,)

    def test_deterministic(self):
        """Ayni kayit her seferinde birebir ayni vektor uretmeli."""
        ext = self._load()
        rec = {
            "eventName": "AssumeRole",
            "sourceIPAddress": "10.0.0.1",
            "awsRegion": "us-east-1",
        }
        v1 = ext.transform_record(rec)
        v2 = ext.transform_record(rec)
        np.testing.assert_array_equal(v1, v2)

    def test_unknown_label(self):
        """Egitimde gorulmemis etiketler -1 donmeli (bilinmeyen kategori)."""
        ext = self._load()
        if not ext.encoders:
            pytest.skip("Encoder bulunamadi")
        col = next(iter(ext.encoders))
        assert ext._encode_value(col, "BILINMEYEN_DEGER_XYZ_99") == -1

    def test_no_nan_in_output(self):
        """Cikti vektorunde NaN olmamali (scaler sonrasi bile)."""
        ext = self._load()
        vec = ext.transform_record({"eventName": "TestEvent"})
        assert not np.isnan(vec).any(), "Cikti vektorunde NaN tespit edildi"


# ===================================================================
#  MODEL TUTARLILIK TESTLERI
# ===================================================================


class TestModelConsistency:
    """Model ciktisinin deterministik ve anlamli oldugunu dogrular."""

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        """Egitilmis model dosyasi yoksa testi atla (CI ortami)."""
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model dosyasi bulunamadi")

    def _load_model(self):
        import torch
        from model import AnomalyLSTMAutoencoder
        from feature_extractor import FeatureExtractor

        ext = FeatureExtractor.load(SCALER_PATH, ENCODER_PATH)
        n = ext.scaler.n_features_in_
        model = AnomalyLSTMAutoencoder(n)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        model.eval()
        return model, n

    def test_same_input_same_output(self):
        """Deterministik cikti -- ayni girdi, ayni sonuc (eval modunda)."""
        import torch

        model, n = self._load_model()
        x = torch.randn(1, SEQUENCE_LENGTH, n)
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        torch.testing.assert_close(y1, y2)

    def test_noise_higher_mse(self):
        """Gurultulu girdi, temiz girdiden daha yuksek MSE vermeli.

        Bu test modelin normal/anormal ayrimi yapabilme kapasitesinin
        temel bir dogrulamasidir.
        """
        import torch

        model, n = self._load_model()

        clean = torch.zeros(1, SEQUENCE_LENGTH, n)
        noisy = torch.randn(1, SEQUENCE_LENGTH, n) * 10

        with torch.no_grad():
            clean_mse = torch.mean(torch.pow(clean - model(clean), 2)).item()
            noisy_mse = torch.mean(torch.pow(noisy - model(noisy), 2)).item()

        assert noisy_mse > clean_mse, (
            f"Gurultu MSE ({noisy_mse:.4f}) temiz MSE'den ({clean_mse:.4f}) "
            f"buyuk olmali"
        )


# ===================================================================
#  ENTEGRASYON TESTLERI -- analyze_records
# ===================================================================


class TestAnalyzeRecords:
    """analyze_records fonksiyonu -- projenin en kritik bileseni.

    Bu testler tam inference pipeline'ini test eder:
    feature extraction -> model inference -> threshold -> defense
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        """Egitilmis model dosyalari yoksa atla."""
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
            pytest.skip("Egitilmis model dosyalari bulunamadi")

    def _setup(self):
        import torch
        from model import AnomalyLSTMAutoencoder
        from feature_extractor import FeatureExtractor

        fe = FeatureExtractor.load(SCALER_PATH, ENCODER_PATH)
        n = fe.scaler.n_features_in_
        device = torch.device("cpu")
        model = AnomalyLSTMAutoencoder(n).to(device)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        model.eval()
        return model, fe, device

    def test_returns_record_count(self, tmp_ban_db):
        """Islenen kayit sayisini dogru donmeli (whitelist + normal dahil)."""
        import main

        model, fe, device = self._setup()
        records = [
            {"eventName": "AssumeRole", "sourceIPAddress": "10.0.0.1"},
            {"eventName": "GetObject", "sourceIPAddress": "10.0.0.2"},
            {"eventName": "PutObject", "sourceIPAddress": "10.0.0.3"},
        ]

        count = main.analyze_records(records, model, fe, 999999.0, device, {})
        assert count == len(records)

    def test_whitelisted_ips_skipped(self, tmp_ban_db):
        """Whitelist'teki IP'ler analiz edilmeden atlanmali."""
        import main
        from config import WHITELIST

        model, fe, device = self._setup()
        wl_ip = next(iter(WHITELIST))
        records = [{"eventName": "Test", "sourceIPAddress": wl_ip}] * 3

        count = main.analyze_records(records, model, fe, 999999.0, device, {})
        assert count == 3

    def test_threat_triggers_ban(self, tmp_ban_db):
        """Esik asildinda ve firewall basariliysa IP ban DB'ye eklenmeli."""
        import main

        model, fe, device = self._setup()
        records = [
            {"eventName": f"Event{i}", "sourceIPAddress": "10.99.99.99"}
            for i in range(SEQUENCE_LENGTH + 1)
        ]

        banned_db = {}
        with patch("main.execute_real_firewall_ban", return_value=True):
            main.analyze_records(records, model, fe, 0.0, device, banned_db)
        assert "10.99.99.99" in banned_db

    def test_already_banned_ip_skipped(self, tmp_ban_db):
        """Zaten banli IP tekrar analiz edilmemeli (gereksiz islem onlemi)."""
        import main

        model, fe, device = self._setup()
        banned_db = {"10.0.0.1": {"timestamp": "2024-01-01", "event": "x", "loss": 1.0}}
        records = [{"eventName": "Test", "sourceIPAddress": "10.0.0.1"}] * 3

        count = main.analyze_records(
            records, model, fe, 999999.0, device, banned_db
        )
        assert count == 3


# ===================================================================
#  SAVUNMA ORKESTRATORU TESTLERI
# ===================================================================


class TestExecuteDefense:
    """execute_defense fonksiyonu -- firewall + ban DB orkestrasyonu."""

    def test_whitelist_ip_not_banned(self):
        """Whitelist'teki IP hicbir kosulda banlanmamali."""
        import main
        from config import WHITELIST

        db = {}
        wl_ip = next(iter(WHITELIST))
        main.execute_defense(wl_ip, "TestEvent", 100.0, 1.0, db)
        assert wl_ip not in db

    def test_invalid_ip_not_banned(self):
        """Gecersiz formattaki IP reddedilmeli (command injection korumasI)."""
        import main

        db = {}
        main.execute_defense("not_an_ip", "TestEvent", 100.0, 1.0, db)
        assert "not_an_ip" not in db

    def test_firewall_called_with_correct_ip(self, tmp_ban_db):
        """Firewall basarili olursa IP ban DB'ye eklenmeli."""
        import main

        db = {}
        with patch("main.execute_real_firewall_ban", return_value=True) as mock_fw:
            main.execute_defense("10.20.30.40", "SuspiciousEvent", 50.0, 1.0, db)
            mock_fw.assert_called_once_with("10.20.30.40")
        assert "10.20.30.40" in db

    def test_firewall_failure_no_ban(self, tmp_ban_db):
        """Firewall basarisiz olursa IP ban DB'ye eklenmemeli.

        Bu test, execute_defense'in yanlis guvenlik algisi
        olusturmadigini dogrular: firewall kurali yazilamamissa
        IP 'banlandi' olarak kaydedilmez.
        """
        import main

        db = {}
        with patch("main.execute_real_firewall_ban", return_value=False):
            main.execute_defense("10.20.30.40", "SuspiciousEvent", 50.0, 1.0, db)
        assert "10.20.30.40" not in db, (
            "Firewall basarisiz oldugunda IP ban DB'ye eklenmemeli"
        )
