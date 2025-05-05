from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional

import torch
from torch import nn, Tensor

# Assuming ops.misc and transforms._presets are available in the project structure
# If not, these imports might need adjustment based on the actual project layout.
# For SqueezeExcitation, let's define a placeholder or use a known implementation if available.
# from ..ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer
# from ..transforms._presets import ImageClassification
# from ..utils import _log_api_usage_once
# from ._api import register_model, Weights, WeightsEnum # Assuming these are not needed as per request
# from ._meta import _IMAGENET_CATEGORIES # Assuming not needed
# from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface # Assuming not needed, implementing _make_divisible locally

# Helper function implementation (originally from ._utils)
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# Simplified Conv2dNormActivation - assuming BatchNorm2d and activation are passed directly
class Conv2dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = None,
        bias: Optional[bool] = None,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

# Squeeze-and-Excitation layer implementation (corrected)
class SElayer(nn.Module):
    # Corrected signature: takes the number of channels it operates on
    def __init__(self, channels: int, reduction: int = 4, scale_activation: Callable[..., nn.Module] = nn.Hardsigmoid):
        super().__init__()
        # Calculate squeeze channels internally
        squeeze_channels = _make_divisible(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, squeeze_channels), # Input features = channels
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_channels, channels), # Output features = channels
            scale_activation() if scale_activation else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size() # c = input channels
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # HS stands for HardSwish
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: list[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            # Pass only the number of channels the SE layer operates on (expanded_channels)
            layers.append(se_layer(cnf.expanded_channels)) # Corrected instantiation

        # project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1 # What is this used for? Maybe related to Conditional Normalization?

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: list[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000, # Default from torchvision, adjust as needed
        input_channels: int = 1, # Added for audio spectrograms
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class adapted for audio.

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure.
            last_channel (int): The number of channels on the penultimate layer.
            num_classes (int): Number of output classes.
            input_channels (int): Number of input channels (e.g., 1 for mono spectrogram).
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block.
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer.
            dropout (float): The dropout probability.
        """
        super().__init__()
        # _log_api_usage_once(self) # Removed dependency

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            # Using BatchNorm2d as default, similar to original implementation
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: list[nn.Module] = []

        # building first layer - adapted for input_channels
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                input_channels, # Use specified input channels
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels # As per original implementation
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
         # Add channel dimension if input is (batch, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1) # Add channel dim -> (batch, 1, height, width)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# Configuration function adapted from torchvision
def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            # input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            # input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel

# Factory functions for creating MobileNetV3 models without pretrained weights
# Removed weights, progress, and related logic as per request

def mobilenet_v3_large(num_classes: int, input_channels: int = 1, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__.
    Modified for custom number of classes and input channels, without pretraining.

    Args:
        num_classes (int): Number of output classes.
        input_channels (int): Number of input channels (e.g., 1 for mono spectrogram).
        **kwargs: Additional parameters passed to the MobileNetV3 base class.
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    # Pass num_classes and input_channels to the MobileNetV3 constructor
    return MobileNetV3(inverted_residual_setting, last_channel, num_classes=num_classes, input_channels=input_channels, **kwargs)


def mobilenet_v3_small(num_classes: int, input_channels: int = 1, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__.
    Modified for custom number of classes and input channels, without pretraining.

    Args:
        num_classes (int): Number of output classes.
        input_channels (int): Number of input channels (e.g., 1 for mono spectrogram).
        **kwargs: Additional parameters passed to the MobileNetV3 base class.
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    # Pass num_classes and input_channels to the MobileNetV3 constructor
    return MobileNetV3(inverted_residual_setting, last_channel, num_classes=num_classes, input_channels=input_channels, **kwargs)

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Example: Create a MobileNetV3 Large model for 50 classes with 1 input channel (spectrogram)
    model_large = mobilenet_v3_large(num_classes=50, input_channels=1)
    print("MobileNetV3 Large (Custom) Architecture:")
    # print(model_large) # This can be very verbose

    # Example: Create a MobileNetV3 Small model
    model_small = mobilenet_v3_small(num_classes=50, input_channels=1)
    print("\nMobileNetV3 Small (Custom) Architecture:")
    # print(model_small)

    # Test with dummy input
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 1, 224, 224) # Batch of 2, 1 channel, 224x224 size
    try:
        output_large = model_large(dummy_input)
        print(f"Large model output shape: {output_large.shape}") # Expected: [2, 50]

        output_small = model_small(dummy_input)
        print(f"Small model output shape: {output_small.shape}") # Expected: [2, 50]
        print("Forward pass successful.")
    except Exception as e:
        print(f"Error during forward pass: {e}")
