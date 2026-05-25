"""Ozellik cikarma ve donusturu modulu.

Egitim ile inference arasinda tutarli veri pipeline'i saglar.
Temel sorumluluklar:
  1. Egitimde: CSV -> LabelEncoder -> StandardScaler -> numpy array
  2. Inference'da: JSON record -> ayni encoder/scaler -> numpy vector
  3. Kaydet/Yukle: joblib ile scaler+encoder serializasyonu

Tasarim karari: Inference sirasinda DataFrame olusturulmaz.
Her kayit dogrudan numpy uzerinde islenir (bellek ve hiz avantaji).

GUVENLIK NOTU: joblib.load pickle deserializasyonu yapar.
Dosya butunlugunu dogrulamak kullanicinin sorumlulugundadir.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import COLUMNS_TO_DROP, TARGET_COL, SCALER_PATH, ENCODER_PATH, MIN_VARIANCE

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Egitim ve inference arasinda tutarli ozellik donusumu.

    Egitimde CSV'den fit edilen LabelEncoder ve StandardScaler,
    inference sirasinda CloudTrail JSON kayitlarina ayni sekilde uygulanir.
    """

    def __init__(self) -> None:
        self.scaler: StandardScaler = StandardScaler()
        self.encoders: dict[str, LabelEncoder] = {}
        self.columns: list[str] = []

    # -----------------------------------------------------------------
    # Egitim
    # -----------------------------------------------------------------

    def fit_transform(self, csv_path: str) -> tuple[np.ndarray, np.ndarray]:
        """CSV verisini oku, encode et, olceklendir."""
        logger.info("Veri yukleniyor: %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)

        existing_drops = [c for c in COLUMNS_TO_DROP if c in df.columns]
        if existing_drops:
            df = df.drop(columns=existing_drops)

        if TARGET_COL in df.columns:
            y = df[TARGET_COL].fillna(0).values.astype(int)
            df = df.drop(columns=[TARGET_COL])
        else:
            y = np.zeros(len(df), dtype=int)

        # Kategorik sutunlari LabelEncoder ile sayisallastir
        self.encoders = {}
        for col in df.columns:
            if (
                df[col].dtype == "object"
                or df[col].dtype.name == "category"
                or df[col].dtype == "bool"
            ):
                df[col] = df[col].fillna("missing").astype(str)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.encoders[col] = le

        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Varyansi sifir veya ihmal edilebilir duzeyde olan sutunlari cikar.
        # Bu sutunlar hic bilgi tasimaz ve scaler'da 0'a bolme yaparak
        # inference sirasinda 10^20 seviyesinde MSE skorlarina yol acar.

        variances = df.var()
        low_var_cols = variances[variances < MIN_VARIANCE].index.tolist()
        if low_var_cols:
            logger.info(
                "Dusuk varyansli sutun cikariliyor: %d/%d (esik: %g)",
                len(low_var_cols),
                len(df.columns),
                MIN_VARIANCE,
            )
            df = df.drop(columns=low_var_cols)
            # Cikarilan sutunlardaki encoder'lari da temizle
            for col in low_var_cols:
                self.encoders.pop(col, None)

        self.columns = list(df.columns)

        X = df.values.astype(np.float32)
        if np.isnan(X).any():
            X = np.nan_to_num(X)

        logger.info("StandardScaler uygulaniyor (%d sutun)...", len(self.columns))
        X_scaled = self.scaler.fit_transform(X)

        variance = float(np.var(X_scaled))
        logger.info("Veri varyansi: %.4f", variance)
        if variance < 0.1:
            logger.warning(
                "Dusuk varyans -- sabit degerler agirliklarin donmasina yol acabilir"
            )

        return X_scaled, y

    # -----------------------------------------------------------------
    # Kaydet / Yukle
    # -----------------------------------------------------------------

    def save(
        self, scaler_path: str | None = None, encoder_path: str | None = None
    ) -> None:
        """Scaler ve encoder paketini diske yaz."""
        scaler_path = scaler_path or SCALER_PATH
        encoder_path = encoder_path or ENCODER_PATH

        joblib.dump(self.scaler, scaler_path)
        logger.info("Scaler kaydedildi: %s", scaler_path)

        bundle = {"encoders": self.encoders, "columns": self.columns}
        joblib.dump(bundle, encoder_path)
        logger.info("Encoder paketi kaydedildi: %s", encoder_path)

    @classmethod
    def load(
        cls, scaler_path: str | None = None, encoder_path: str | None = None
    ) -> "FeatureExtractor":
        """Diskten yukleyerek hazir FeatureExtractor dondur."""
        scaler_path = scaler_path or SCALER_PATH
        encoder_path = encoder_path or ENCODER_PATH

        ext = cls()
        ext.scaler = joblib.load(scaler_path)
        bundle = joblib.load(encoder_path)
        ext.encoders = bundle["encoders"]
        ext.columns = bundle["columns"]
        return ext

    # -----------------------------------------------------------------
    # Inference -- DataFrame olusturmadan direkt numpy uzerinde calisir
    # -----------------------------------------------------------------

    def transform_record(self, record: dict[str, Any]) -> np.ndarray:
        """CloudTrail JSON kaydini ozellik vektorune donustur."""
        n = len(self.columns)
        vector = np.zeros(n, dtype=np.float32)

        for i, col in enumerate(self.columns):
            raw = self._extract_value(col, record)

            if col in self.encoders:
                vector[i] = self._encode_value(col, raw)
            else:
                try:
                    val = float(raw)
                    vector[i] = val if not np.isnan(val) else 0.0
                except (ValueError, TypeError):
                    vector[i] = 0.0

        return self.scaler.transform(vector.reshape(1, -1))[0]

    @staticmethod
    def _extract_value(column_name: str, record: dict[str, Any]) -> Any:
        """CSV sutun adini JSON'daki nested yapiya esle.

        'CloudTrailEvent.X.Y.Z' -> record['X']['Y']['Z']
        """
        prefix = "CloudTrailEvent."
        path = (
            column_name[len(prefix) :]
            if column_name.startswith(prefix)
            else column_name
        )

        current: Any = record
        for key in path.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return np.nan
        return current

    def _encode_value(self, col: str, value: Any) -> int:
        """LabelEncoder ile kodla.

        Bilinmeyen etiketler -1 doner. Scaler bu degeri negatif yone
        kaydirir, bu da anomali skoruna etki eder -- kasitli tercih.
        """
        le = self.encoders[col]
        label = str(value) if not pd.isna(value) else "missing"
        if label in le.classes_:
            return int(le.transform([label])[0])
        return -1
