from typing import Sequence, overload

from .openai import OpenAIUNet, UNet


@overload
def build(in_channels: int, out_channels: int, /, *, attention_resolutions: Sequence[int] = [32, 16, 8], channel_mult: Sequence[int] = [1, 4, 8]) -> OpenAIUNet: ...

@overload
def build(in_channels: int, out_channels: int, /, *, attention_resolutions: Sequence[int] = [32, 16, 8], channel_mult: Sequence[int] = [1, 4, 8], use_time_wrap: bool = False) -> UNet: ...

@overload
def build(in_channels: int, out_channels: int, /, *, attention_resolutions: Sequence[int] = [32, 16, 8], channel_mult: Sequence[int] = [1, 4, 8], use_time_wrap: bool = False) -> OpenAIUNet: ...

def build(in_channels: int, out_channels: int, /, *, attention_resolutions: Sequence[int] = [32, 16, 8], channel_mult: Sequence[int] = [1, 4, 8], use_time_wrap: bool = False) -> OpenAIUNet:
    return UNet(in_channels, 128, out_channels, num_res_blocks=2, attention_resolutions=attention_resolutions, channel_mult=channel_mult, num_heads=8, num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True) if use_time_wrap else OpenAIUNet(in_channels, 128, out_channels, num_res_blocks=2, attention_resolutions=attention_resolutions, channel_mult=channel_mult, num_heads=8, num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True)

def build_unet(in_channels: int, out_channels: int, /, *, attention_resolutions: Sequence[int] = [32, 16, 8], channel_mults: Sequence[int] = [1, 4, 8]) -> UNet:
    return build(in_channels, out_channels, attention_resolutions=attention_resolutions, channel_mult=channel_mults, use_time_wrap=True)
