"""LSTM Autoencoder -- anomali tespiti icin tasarlanmis sinir agi mimarisi.

Encoder-bottleneck-decoder yapisi ile girdi dizisini sikistirip yeniden
yapilandirir. Normal trafik icin dusuk reconstruction hatasi, anomali
trafik icin yuksek hata uretmesi beklenir.

Mimari kararlarin gerekceleri:
  - LSTM: CloudTrail kayitlarinin zamansal bagimliligini yakalamak icin.
  - LeakyReLU: Dead neuron sorununu onlemek icin (ReLU yerine).
  - Tek katman LSTM: 17K kayitlik veri seti icin yeterli kapasite.
    Cok katmanli yapida overfitting riski artar.
"""

import torch.nn as nn
from torch import Tensor


class AnomalyLSTMAutoencoder(nn.Module):
    """Zaman dizisi tabanli LSTM Autoencoder.

    Parametreler:
        input_dim: Ozellik vektoru boyutu (egitimde feature_extractor belirler).
        hidden_dim: LSTM gizli katman boyutu. Buyuk deger = daha fazla kapasite
                     ama daha fazla overfitting riski.
        bottleneck_dim: Darbogazdaki boyut. Bilgi sikistirma oranini belirler.

    Forward cikti boyutu girdiyle ayni: (batch, seq_len, input_dim).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=1, batch_first=True
        )
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        self.expand = nn.Linear(bottleneck_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.1)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=1, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Girdi dizisini encode-decode ederek yeniden yapilandir."""
        enc, _ = self.encoder_lstm(x)
        z = self.activation(self.bottleneck(enc))
        z = self.activation(self.expand(z))
        dec, _ = self.decoder_lstm(z)
        return self.output_layer(dec)
