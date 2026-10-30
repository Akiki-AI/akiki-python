"""Supported precision settings for VAD model inference and its FFT."""

from enum import Enum


class ModelPrecision(Enum):
    """VAD model precision; reduced precision requires the Rust engine."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    INT8_FP32_INPUT = "int8-fp32-input"

    def __str__(self) -> str:
        return self.value


class FFTPrecision(Enum):
    """VAD FFT precision. Only FP32 is currently supported."""

    FP32 = "fp32"

    def __str__(self) -> str:
        return self.value
