"""Akiki command-line tools: ``akiki vad`` for speech, ``akiki minilm`` for text."""
from dataclasses import asdict
import argparse
import json

from . import __version__


def main():
    """Dispatch Akiki subcommands and report command errors with exit status 2.

    The ``akiki`` command and ``python -m akiki`` share this entry point.
    ``akiki vad AUDIO`` prints one JSON result per complete audio frame.
    ``akiki minilm --model PATH TEXT`` prints a classification or embedding as JSON.
    Use ``akiki --help`` to list commands and ``akiki <command> --help`` for
    its options. The root command and every subcommand support ``--version``.
    """
    from ._quantization import MODEL_PRECISIONS
    from .minilm import MODEL_PRECISIONS as MINILM_PRECISIONS
    parser = argparse.ArgumentParser(prog="akiki", description="Akiki command-line tools.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, title="commands")
    vad = commands.add_parser("vad", help="Detect speech in mono 16 kHz audio",
                              description="Run Akiki VAD locally on mono 16 kHz audio.")
    vad.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    vad.add_argument("audio", help="Input mono 16 kHz audio file")
    vad.add_argument("--threshold", type=float, default=None,
                     help="Speech probability threshold (default: model setting)")
    vad.add_argument("--engine", choices=["auto", "rust", "onnx", "pytorch"], default="auto",
                     help="Inference engine (default: auto)")
    vad.add_argument("--library-path", help="Path to a local Rust VAD shared library")
    vad.add_argument("--model-precision", choices=MODEL_PRECISIONS, default=None,
                     help="Model precision (default: INT8 for Rust, FP32 for ONNX/PyTorch)")
    vad.add_argument("--fft-precision", choices=["fp32"], default="fp32",
                     help="FFT precision (fixed at fp32)")
    vad.add_argument("--cache-dir", help="Directory for locally generated model variants")
    vad.set_defaults(handler=_run_vad)
    minilm = commands.add_parser("minilm", help="Embed or classify text with a MiniLM model",
                                 description="Run a MiniLM text model locally: a plain checkpoint or an "
                                             "Akiki Ariadne variant. Model weights are not bundled; pass a "
                                             "local copy with --model.")
    minilm.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    minilm.add_argument("text", help="Text in the model's input format")
    minilm.add_argument("--model", required=True,
                        help="Checkpoint folder, or a variant folder with base_model/ and adapter.safetensors")
    minilm.add_argument("--adapter", help="Adapter safetensors file (default: the variant folder's)")
    minilm.add_argument("--embedding", action="store_true",
                        help="Include the embedding (always printed for models without a head)")
    minilm.add_argument("--chunk-tokens", type=int, default=None,
                        help="Split long text into chunks of at most this many tokens")
    minilm.add_argument("--threads", type=int, default=None, help="CPU worker threads (default: up to 4)")
    minilm.add_argument("--model-precision", choices=MINILM_PRECISIONS, default="int8",
                        help="Weight and projection precision (default: int8)")
    minilm.add_argument("--library-path", help="Path to a local Rust MiniLM shared library")
    minilm.set_defaults(handler=_run_minilm)
    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{parser.prog} {args.command}: {error}\n")


def _run_vad(args):
    """Print frame decisions for a local audio file using the requested engine."""
    import soundfile as sf
    from . import VAD
    with VAD(threshold=args.threshold, engine=args.engine, library_path=args.library_path,
             model_precision=args.model_precision, fft_precision=args.fft_precision,
             cache_dir=args.cache_dir) as detector, sf.SoundFile(args.audio) as source:
        if source.samplerate != 16000 or source.channels != 1:
            raise ValueError("Expected mono 16 kHz audio")
        for block in source.blocks(blocksize=16000, dtype="float32"):
            for frame in detector.push(block):
                print(json.dumps(asdict(frame)))


def _run_minilm(args):
    """Print the classification of one text for models with a head, else its embedding."""
    from . import MiniLM
    with MiniLM(args.model, args.adapter, threads=args.threads, model_precision=args.model_precision,
                chunk_tokens=args.chunk_tokens, library_path=args.library_path) as model:
        result = asdict(model.classify(args.text)) if model.head_size else {}
        if args.embedding or not model.head_size:
            result["embedding"] = model.embed(args.text).tolist()
        print(json.dumps(result))


if __name__ == "__main__":
    main()
