"""Command-line interface for local speech detection in mono, 16 kHz files."""
from dataclasses import asdict
import argparse
import json

from . import __version__


def main():
    """Parse CLI options and print one JSON result per complete audio frame.

    The ``akiki`` command and ``python -m akiki`` share this entry point.
    ``--version`` prints the installed package version. Invalid arguments,
    unsupported audio, or engine errors terminate with exit status 2.
    """
    from ._quantization import MODEL_PRECISIONS
    parser = argparse.ArgumentParser(description="Run Akiki VAD locally on mono 16 kHz audio.")
    parser.add_argument("--version", action="version", version=f"akiki {__version__}")
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--engine", choices=["auto", "rust", "onnx", "pytorch"], default="auto")
    parser.add_argument("--library-path", help="Path to a local Rust VAD shared library")
    parser.add_argument("--model-precision", choices=MODEL_PRECISIONS, default=None,
                        help="Model precision (default: INT8 for Rust, FP32 for ONNX/PyTorch)")
    parser.add_argument("--fft-precision", choices=["fp32"], default="fp32")
    parser.add_argument("--cache-dir", help="Directory for locally generated model variants")
    args = parser.parse_args()
    import soundfile as sf
    from . import VAD
    try:
        with VAD(threshold=args.threshold, engine=args.engine, library_path=args.library_path,
                 model_precision=args.model_precision, fft_precision=args.fft_precision,
                 cache_dir=args.cache_dir) as detector, sf.SoundFile(args.audio) as source:
            if source.samplerate != 16000 or source.channels != 1:
                raise ValueError("Expected mono 16 kHz audio")
            for block in source.blocks(blocksize=16000, dtype="float32"):
                for frame in detector.push(block):
                    print(json.dumps(asdict(frame)))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"akiki: {error}\n")


if __name__ == "__main__":
    main()
