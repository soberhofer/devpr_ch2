import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioMLP(nn.Module):
    def __init__(self, n_steps, n_mels, hidden1_size, hidden2_size, output_size, time_reduce=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_reduce = time_reduce
        # optimized for GPU, faster than x.reshape(*x.shape[:-1], -1, 2).mean(-1)
        self.pool = nn.AvgPool1d(kernel_size=time_reduce, stride=time_reduce)  # Non-overlapping averaging

        self.fc1 = nn.Linear(n_steps * n_mels, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # reduce time dimension
        shape = x.shape
        x = x.reshape(-1, 1, x.shape[-1])
        x = self.pool(x)  # (4096, 1, 431//n)
        x = x.reshape(shape[0], shape[1], shape[2], -1)

        # 2D to 1D
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AudioCNN(nn.Module):
    def __init__(self, n_mels, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (n_mels // 4) * (431 // 4), 128) # Assuming input is 431
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


import torch
import torch.nn as nn

class TFCNN(nn.Module):
    def __init__(self, num_classes=50, in_channels=1):
        super(TFCNN, self).__init__()
        
        # Time-Frequency Convolutional Blocks
        self.features = nn.Sequential(
            # Block 1: Temporal-Conv (wider in frequency dimension)
            nn.Conv2d(in_channels, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4)),
            
            # Block 2: Balanced Conv
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 4)),
            
            # Block 3: Square Conv
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class HPSS(nn.Module):
    def __init__(self, kernel_size=17, n_mels=128, output_size=50):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(4,4), stride=(4,4))
        self.fc1 = nn.Linear(32 * (n_mels // 4) * (431 // 4), 128) # Assuming input is 431
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        # Approximate HPSS using average pooling
        harmonic = F.avg_pool2d(x, kernel_size=(1, self.kernel_size), stride=1, padding=(0, self.pad))
        percussive = F.avg_pool2d(x, kernel_size=(self.kernel_size, 1), stride=1, padding=(self.pad, 0))
        mask_h = (harmonic > percussive).float()
        mask_p = 1 - mask_h
        harmonic_masked = x * mask_h
        percussive_masked = x * mask_p
        return harmonic_masked, percussive_masked

class FrequencyAttention(nn.Module):
    def __init__(self, in_channels=1, T=64):
        super().__init__()
        layers = []
        for i in range(6):
            in_ch = in_channels if i == 0 else 32
            layers += [
                nn.Conv2d(in_ch, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ]
        layers.append(nn.Conv2d(32, 1, kernel_size=1))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        x = self.convs(x)  # (batch, 1, F, 1)
        x = F.softmax(x, dim=-1)
        x = x.transpose(2,3) # (batch, 1, F)
        x = x.unsqueeze(1)
        x = F.interpolate(x, size=(431,), mode='linear', align_corners=False)
        x = x.transpose(2,3)
        return x

class TemporalAttention(nn.Module):
    def __init__(self, in_channels=1, F=128):
        super().__init__()
        layers = []
        for i in range(6):
            in_ch = in_channels if i == 0 else 32
            layers += [
                nn.Conv2d(in_ch, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ]
        layers.append(nn.Conv2d(32, 1, kernel_size=1))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        x = self.convs(x)  # (batch, 1, 1, T)
        x = self.convs(x)  # (batch, 1, 1, T)
        return F.softmax(x, dim=-1)

class TFCNN2(nn.Module):
    def __init__(self, num_classes, F=128, T=64):
        super().__init__()
        self.hpss = HPSS()
        self.freq_attn = FrequencyAttention(T=T)
        self.temp_attn = TemporalAttention(F=F)
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))  # For alpha and beta

        # Backbone CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), stride=2, padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(5, 3), stride=2, padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Dynamically calculate FC input size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, F, T)
            dummy = self.backbone(dummy)
            fc_in = dummy.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        harmonic, percussive = self.hpss(x)
        F_weights = self.freq_attn(harmonic)  # (B, 1, F)
        T_weights = self.temp_attn(percussive)  # (B, 1, T)
        print(f"harmonic shape: {harmonic.shape}")
        print(f"F_weights shape: {F_weights.shape}")
        
        S_F = harmonic * F_weights.unsqueeze(-1)  # (B,1,F,T)
        S_T = percussive * T_weights.unsqueeze(2)   # (B,1,F,T)
        
        # Combine with learned weights
        alpha, beta = F.softmax(self.weights, dim=0)
        combined = alpha * S_T + beta * S_F
        
        features = self.backbone(combined)
        return self.fc(features)
