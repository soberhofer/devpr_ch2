import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2Audio(nn.Module):
    def __init__(self, num_classes, pretrained=False, input_channels=1, dropout_prob=0.5):
        """
        MobileNetV2 adapted for audio spectrogram classification.

        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use pretrained weights from ImageNet.
            input_channels (int): Number of input channels (usually 1 for spectrograms).
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Load MobileNetV2, optionally with pretrained weights
        if pretrained:
            self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            self.mobilenet = models.mobilenet_v2(weights=None)

        # Modify the first convolutional layer to accept the specified number of input channels
        # Original first layer: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_first_layer = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias
        )

        # If pretrained and input channels changed, handle weight initialization (optional, simple approach here)
        # A more sophisticated approach might involve averaging the original weights or specific initialization.
        if pretrained and self.input_channels != 3:
             # Initialize new weights, keeping others pretrained
             # Simple initialization: Kaiming normal for the new conv layer
             nn.init.kaiming_normal_(self.mobilenet.features[0][0].weight, mode='fan_out', nonlinearity='relu')
             if self.mobilenet.features[0][0].bias is not None:
                 nn.init.constant_(self.mobilenet.features[0][0].bias, 0)
             print(f"Warning: Initialized first conv layer for {self.input_channels} channels. Other layers remain pretrained.")


        # Modify the final classifier layer
        # Original classifier: Linear(in_features=1280, out_features=1000, bias=True)
        # Add dropout layer
        self.mobilenet.classifier[0] = nn.Dropout(p=dropout_prob)
        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor, expected shape (batch_size, channels, height, width)
                             or (batch_size, channels, n_mels, time_steps).
                             Ensure the input tensor has the channel dimension.
        """
        # Add channel dimension if input is (batch, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1) # Add channel dim -> (batch, 1, height, width)

        # Ensure input has the correct number of channels expected by the modified first layer
        if x.shape[1] != self.input_channels:
             # This might happen if the dataset provides multi-channel data unexpectedly
             # Or if the input wasn't unsqueezed correctly.
             # Simple fix: If input is mono and model expects >1 channel, repeat the channel.
             if x.shape[1] == 1 and self.input_channels > 1:
                 x = x.repeat(1, self.input_channels, 1, 1)
                 # print(f"Warning: Repeating input channel to match model's expected {self.input_channels} channels.")
             else:
                 # If mismatch is more complex, raise an error.
                 raise ValueError(f"Input tensor has {x.shape[1]} channels, but model expects {self.input_channels}")

        return self.mobilenet(x)
