"""LSTM Autoencoder egitim pipeline'i.

Tam egitim dongusu: veri hazirlama -> model olusturma -> egitim -> degerlendirme -> kayit.

Tasarim kararlari:
  - Sentetik anomali: Test setine Gaussian gurultu ekleyerek modelin
    anomali tespiti kapasitesi olculur. Gercek saldiri kaliplarina
    degil, reconstruction hatasina dayali bir degerlendirmedir.
  - Dinamik esik: val_mse_mean + 3*std. Sabit esik yerine veriye
    adapte olan esik, farkli dagilimlar icin daha saglamdir.
  - Seed sabitleme: torch, numpy ve random uclugu ayni SEED ile
    baslatilarak egitimden egitme tekrarlanabilirlik saglanir.
"""

import logging
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)

from config import (
    DATASET_PATH,
    SEQUENCE_LENGTH,
    NORMAL_CLASS,
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    THRESHOLD_PATH,
    setup_logging,
    SEED,
)

from model import AnomalyLSTMAutoencoder
from feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


def create_sequences(
    data: np.ndarray, labels: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Kayan pencere ile zaman dizileri olustur."""
    xs, ys = [], []
    for i in range(len(data) - seq_len + 1):
        xs.append(data[i : i + seq_len])
        ys.append(labels[i + seq_len - 1])
    return np.array(xs), np.array(ys)


class CloudAITrainer:
    """LSTM Autoencoder egitim pipeline'i.

    Kullanim:
        trainer = CloudAITrainer(data_path="...", epochs=120)
        trainer.prepare_data()
        trainer.build_and_train()
        trainer.evaluate()
        trainer.export_assets()

    Her adim bagimsiz cagrilabilir (debug icin) ama sirali cagri beklenir.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = SEQUENCE_LENGTH,
        batch_size: int = 256,
        epochs: int = 40,
        lr: float = 1e-3,
    ) -> None:
        self.data_path = data_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.feature_extractor = FeatureExtractor()
        self.model: AnomalyLSTMAutoencoder | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold: float = 0.0

        self.X_test: np.ndarray = np.array([])
        self.y_test: np.ndarray = np.array([])
        self.X_test_tensor: torch.Tensor = torch.empty(0)
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.input_dim: int = 0

    def prepare_data(self) -> None:
        """Veriyi oku, donustur, dizilere ayir."""
        X_scaled, y = self.feature_extractor.fit_transform(self.data_path)

        logger.info("Zaman dizileri olusturuluyor (seq=%d)", self.seq_len)
        X_seq, y_seq = create_sequences(X_scaled, y, self.seq_len)

        X_tv, X_test, y_tv, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=SEED
        )
        X_train, X_val = train_test_split(X_tv, test_size=0.15, random_state=SEED)

        # Sentetik anomali -- Gaussian gurultu.
        # Gercek saldiri kaliplarindan farkli, sadece reconstruction
        # hatasini test etmek icin kullanilir.
        n_anom = int(len(X_test) * 0.15)
        rng = np.random.RandomState(SEED)
        synth = X_test[:n_anom].copy() + rng.normal(1.5, 2.0, X_test[:n_anom].shape)
        X_test = np.vstack([X_test, synth])
        y_test = np.concatenate([y_test, np.ones(n_anom, dtype=int)])

        idx = rng.permutation(len(X_test))
        self.X_test = X_test[idx]
        self.y_test = y_test[idx]

        self.train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        self.input_dim = X_train.shape[2]
        logger.info("Hazirlik tamam. Girdi boyutu: %d", self.input_dim)

    def build_and_train(self) -> None:
        """Modeli olustur ve egit."""
        logger.info("Egitim basliyor (%s, %d epoch)", self.device, self.epochs)
        self.model = AnomalyLSTMAutoencoder(self.input_dim).to(self.device)
        assert self.model is not None
        assert self.train_loader is not None
        assert self.val_loader is not None

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for (inputs,) in self.train_loader:
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(inputs), inputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(self.train_loader.dataset)  # type: ignore

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (inputs,) in self.val_loader:
                    inputs = inputs.to(self.device)
                    loss = criterion(self.model(inputs), inputs)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(self.val_loader.dataset)  # type: ignore
            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Epoch [%02d/%d] train=%.6f val=%.6f",
                    epoch + 1,
                    self.epochs,
                    train_loss,
                    val_loss,
                )

    def evaluate(self) -> None:
        """Test seti uzerinde degerlendir ve esik hesapla."""
        logger.info("Degerlendirme yapiliyor...")
        assert self.model is not None
        assert self.val_loader is not None
        assert self.X_test_tensor is not None
        self.model.eval()

        with torch.no_grad():
            recon = self.model(self.X_test_tensor)
            mse = (
                torch.mean(torch.pow(self.X_test_tensor - recon, 2), dim=(1, 2))
                .cpu()
                .numpy()
            )

            val_in = self.val_loader.dataset.tensors[0].to(self.device)  # type: ignore
            val_recon = self.model(val_in)
            val_mse = (
                torch.mean(torch.pow(val_in - val_recon, 2), dim=(1, 2)).cpu().numpy()
            )

        # Esik: validation ortalamasindan 3 standart sapma yukari
        self.threshold = float(np.mean(val_mse) + 3 * np.std(val_mse))
        logger.info("Dinamik esik: %.6f", self.threshold)

        preds = (mse > self.threshold).astype(int)
        logger.info("--- MODEL RAPORU ---")
        logger.info(
            "F1=%.4f  Precision=%.4f  Recall=%.4f  AUC=%.4f",
            f1_score(self.y_test, preds),
            precision_score(self.y_test, preds),
            recall_score(self.y_test, preds),
            roc_auc_score(self.y_test, preds),
        )
        logger.info("Confusion Matrix:\n%s", confusion_matrix(self.y_test, preds))

    def export_assets(self) -> None:
        """Model, scaler, encoder ve esik degerini kaydet."""
        logger.info("Model varliklari kaydediliyor...")
        assert self.model is not None
        torch.save(self.model.state_dict(), MODEL_PATH)
        self.feature_extractor.save(SCALER_PATH, ENCODER_PATH)

        with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
            json.dump({"threshold": self.threshold}, f, indent=4)

        logger.info(
            "Kaydedilen dosyalar: %s, %s, %s, %s",
            MODEL_PATH,
            SCALER_PATH,
            ENCODER_PATH,
            THRESHOLD_PATH,
        )


if __name__ == "__main__":
    setup_logging()

    # Reproducibility icin tum seed'leri sabitle
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    logger.info("=== CLOUD AI V2 EGITIM MOTORU ===")

    trainer = CloudAITrainer(
        data_path=DATASET_PATH, epochs=120, batch_size=256, lr=1e-3
    )
    try:
        trainer.prepare_data()
        trainer.build_and_train()
        trainer.evaluate()
        trainer.export_assets()
    except Exception as e:
        logger.exception("Kritik hata: %s", e)
