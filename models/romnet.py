import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1  # für BasicBlock ist expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_prob=0.2):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        # Dropout Layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.dropout(out) # Dropout after first relu
        out = self.bn2(self.conv2(out))
        # Dropout after second batchnorm, before adding identity.
        # Original ResNet doesn't typically have dropout here, but following user's BasicBlock.
        out = self.dropout(out) 
        out += identity
        return self.relu(out)

# Renamed OwnNet to Romnet as requested
class Romnet(nn.Module):
    def __init__(self, num_classes=50, dropout_prob=0.2, channels=1): # Added channels parameter for consistency
        super(Romnet, self).__init__()
        self.in_channels = 64

        # Initialer Layer
        # Use the 'channels' parameter for the first convolution's input channels
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Blocks
        self.layer1 = self._make_layer(64, 2, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(128, 2, stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(256, 2, stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(512, 2, stride=2, dropout_prob=dropout_prob)

        # Klassifikationskopf
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Dropout after Fully Connected Layer
        self.dropout_fc = nn.Dropout(dropout_prob) # Renamed to avoid conflict with BasicBlock's dropout

    def _make_layer(self, out_channels, blocks, stride=1, dropout_prob=0.2):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample, dropout_prob=dropout_prob))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, dropout_prob=dropout_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x)) # Original had F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout_fc(x) # Use the renamed dropout layer
        return x
