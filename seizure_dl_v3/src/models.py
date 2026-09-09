"""
Three EEG-specific deep architectures for binary seizure detection.

References:
- EEGNet: Lawhern et al. (2018), J. Neural Eng. 15:056013
- ShallowConvNet: Schirrmeister et al. (2017), Hum. Brain Mapp. 38:5391
- CNN-LSTM: hybrid spatial-temporal architecture

All models accept input shape: (batch, n_channels, n_timesteps)
All models output a single logit: (batch, 1) — to be passed through sigmoid
externally for probability extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    Compact EEG-specific CNN. Lawhern et al. 2018.

    Architecture:
      1. Temporal conv (F1 filters, kernel_length samples wide)
      2. Depthwise spatial conv (D filters per channel, n_channels wide)
      3. Separable conv (F2 pointwise filters)
      4. Linear classifier
    """

    def __init__(self, n_channels, n_samples, F1=8, D=2, F2=16,
                 kernel_length=64, dropout=0.5):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: temporal then spatial
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length),
            padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.conv_depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv
        self.conv_separable_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16),
            padding=(0, 8), groups=F1 * D, bias=False
        )
        self.conv_separable_point = nn.Conv2d(
            F1 * D, F2, (1, 1), bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.conv_temporal(dummy)
            x = self.bn1(x)
            x = self.conv_depthwise(x)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.pool1(x)
            x = self.conv_separable_depth(x)
            x = self.conv_separable_point(x)
            x = self.bn3(x)
            x = F.elu(x)
            x = self.pool2(x)
            self.flatten_size = x.view(1, -1).shape[1]

        self.classifier = nn.Linear(self.flatten_size, 1)

    def forward(self, x):
        # x: (batch, n_channels, n_samples) → (batch, 1, n_channels, n_samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = self.conv_depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # logits


class ShallowConvNet(nn.Module):
    """
    Schirrmeister et al. 2017 — shallow band-power-style EEG CNN.
    The key trick is the square→log activation pair which encodes
    band-power features within an end-to-end network.
    """

    def __init__(self, n_channels, n_samples,
                 n_filters_time=40, filter_time_length=25,
                 n_filters_spat=40, pool_time_length=75,
                 pool_time_stride=15, dropout=0.5):
        super().__init__()
        self.conv_time = nn.Conv2d(
            1, n_filters_time, (1, filter_time_length), bias=True
        )
        self.conv_spat = nn.Conv2d(
            n_filters_time, n_filters_spat, (n_channels, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.conv_time(dummy)
            x = self.conv_spat(x)
            x = self.bn(x)
            x = x ** 2
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            self.flatten_size = x.view(1, -1).shape[1]

        self.classifier = nn.Linear(self.flatten_size, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = x ** 2  # square nonlinearity
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # log nonlinearity
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CNNLSTM(nn.Module):
    """
    Spatial CNN encoder + temporal LSTM aggregator.

    Architecture:
      1. 1D conv across channels (spatial mixing)
      2. Two stacked 1D convs across time
      3. Bidirectional LSTM (2 layers)
      4. Linear classifier on last hidden state
    """

    def __init__(self, n_channels, n_samples,
                 cnn_filters=32, cnn_kernel=7,
                 lstm_hidden=64, lstm_layers=2, dropout=0.4):
        super().__init__()
        # Spatial convolution: n_channels → cnn_filters
        self.spatial_conv = nn.Conv1d(n_channels, cnn_filters, 1)
        self.bn_sp = nn.BatchNorm1d(cnn_filters)

        # Two temporal convolutions
        self.temp_conv1 = nn.Conv1d(
            cnn_filters, cnn_filters, cnn_kernel,
            padding=cnn_kernel // 2
        )
        self.bn_t1 = nn.BatchNorm1d(cnn_filters)
        self.pool1 = nn.MaxPool1d(4)

        self.temp_conv2 = nn.Conv1d(
            cnn_filters, cnn_filters * 2, cnn_kernel,
            padding=cnn_kernel // 2
        )
        self.bn_t2 = nn.BatchNorm1d(cnn_filters * 2)
        self.pool2 = nn.MaxPool1d(4)

        self.drop_cnn = nn.Dropout(dropout)

        # LSTM expects (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        # x: (batch, n_channels, n_samples)
        x = self.spatial_conv(x)
        x = self.bn_sp(x)
        x = F.relu(x)

        x = self.temp_conv1(x)
        x = self.bn_t1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.temp_conv2(x)
        x = self.bn_t2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.drop_cnn(x)

        # (batch, features, time) → (batch, time, features) for LSTM
        x = x.transpose(1, 2)
        out, (h, c) = self.lstm(x)
        # Use last time-step output (concatenated forward + backward)
        last = out[:, -1, :]
        return self.classifier(last)


def build_model(name, n_channels, n_samples, **cfg):
    """Factory function — keeps run scripts model-agnostic."""
    name = name.lower()
    if name == "eegnet":
        return EEGNet(n_channels, n_samples, **cfg)
    elif name == "shallowconvnet":
        return ShallowConvNet(n_channels, n_samples, **cfg)
    elif name == "cnnlstm":
        return CNNLSTM(n_channels, n_samples, **cfg)
    else:
        raise ValueError(f"Unknown model: {name}")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test — verify all three models run
    n_ch, n_t = 18, 1024  # 18 channels, 4s @ 256Hz

    for name in ["eegnet", "shallowconvnet", "cnnlstm"]:
        model = build_model(name, n_ch, n_t)
        x = torch.randn(8, n_ch, n_t)
        y = model(x)
        n_params = count_parameters(model)
        print(f"{name:18s}: input={tuple(x.shape)}, output={tuple(y.shape)}, "
              f"params={n_params:,}")
