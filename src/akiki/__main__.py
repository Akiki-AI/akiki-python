"""Akiki tools for speech, text, and explicit Hugging Face model downloads."""

import argparse
import json
from dataclasses import asdict

from . import Engine, FFTPrecision, ModelPrecision, __version__


def main():
    """Dispatch Akiki subcommands and report command errors with exit status 2.

    The ``akiki`` command and ``python -m akiki`` share this entry point.
    ``akiki vad AUDIO`` prints one JSON result per complete audio frame.
    ``akiki minilm --model PATH TEXT`` prints a classification or embedding as JSON.
    ``akiki download REPO_ID`` fetches a compatible MiniLM model and prints its path.
    Use ``akiki --help`` to list commands and ``akiki <command> --help`` for
    its options. The root command and every subcommand support ``--version``.
    """
    from .minilm import MODEL_PRECISIONS as MINILM_PRECISIONS

    parser = argparse.ArgumentParser(prog="akiki", description="Akiki command-line tools.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, title="commands")
    vad = commands.add_parser(
        "vad",
        help="Detect speech in mono 16 kHz audio",
        description="Run Akiki VAD locally on mono 16 kHz audio.",
    )
    vad.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    vad.add_argument("audio", help="Input mono 16 kHz audio file")
    vad.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Speech probability threshold (default: model setting)",
    )
    vad.add_argument(
        "--engine",
        type=Engine,
        choices=list(Engine),
        default=Engine.AUTO,
        help="Inference engine (default: auto)",
    )
    vad.add_argument("--library-path", help="Path to a local Rust VAD shared library")
    vad.add_argument(
        "--model-precision",
        type=ModelPrecision,
        choices=list(ModelPrecision),
        default=None,
        help="Model precision (default: INT8 for Rust, FP32 for ONNX/PyTorch)",
    )
    vad.add_argument(
        "--fft-precision",
        type=FFTPrecision,
        choices=list(FFTPrecision),
        default=FFTPrecision.FP32,
        help="FFT precision (fixed at fp32)",
    )
    vad.add_argument("--cache-dir", help="Directory for locally generated model variants")
    vad.set_defaults(handler=_run_vad)
    minilm = commands.add_parser(
        "minilm",
        help="Embed or classify text with a MiniLM model",
        description="Run a MiniLM text model locally: a plain checkpoint or an "
        "Akiki Ariadne variant. Pass a local copy with --model, or explicitly "
        "fetch a Hugging Face model with --repo (requires akiki[huggingface]).",
    )
    minilm.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    minilm.add_argument("text", help="Text in the model's input format")
    models = minilm.add_mutually_exclusive_group(required=True)
    models.add_argument(
        "--model",
        help="Checkpoint folder, or a variant folder with base_model/ and adapter.safetensors",
    )
    models.add_argument("--repo", help="Hugging Face model repository to fetch and load")
    _hub_arguments(minilm)
    minilm.add_argument(
        "--adapter", help="Adapter safetensors file (default: the variant folder's)"
    )
    minilm.add_argument(
        "--embedding",
        action="store_true",
        help="Include the embedding (always printed for models without a head)",
    )
    minilm.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="Split long text into chunks of at most this many tokens",
    )
    minilm.add_argument(
        "--threads", type=int, default=None, help="CPU worker threads (default: up to 4)"
    )
    minilm.add_argument(
        "--model-precision",
        choices=MINILM_PRECISIONS,
        default="int8",
        help="Weight and projection precision (default: int8)",
    )
    minilm.add_argument("--library-path", help="Path to a local Rust MiniLM shared library")
    minilm.set_defaults(handler=_run_minilm)
    download = commands.add_parser(
        "download",
        help="Download a MiniLM model from Hugging Face",
        description="Fetch a compatible MiniLM checkpoint without loading an inference engine. "
        "Requires akiki[huggingface]. Prints the local model directory.",
    )
    download.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    download.add_argument("repo_id", help="Hugging Face model repository")
    _hub_arguments(download)
    download.set_defaults(handler=_run_download)
    args = parser.parse_args()
    if args.command == "minilm" and args.model is not None:
        if args.local_files_only or any(
            value is not None for value in (args.revision, args.cache_dir, args.local_dir)
        ):
            parser.error("Hugging Face options require --repo instead of --model")
    try:
        args.handler(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{parser.prog} {args.command}: {error}\n")


def _hub_arguments(parser):
    """Add options shared by explicit Hugging Face commands."""
    parser.add_argument("--revision", help="Hub branch, tag, or full commit hash (default: main)")
    parser.add_argument("--cache-dir", help="Hugging Face Hub cache directory")
    parser.add_argument("--local-dir", help="Write the downloaded model into this directory")
    parser.add_argument(
        "--local-files-only", action="store_true", help="Use cached files without network access"
    )


def _run_download(args):
    """Fetch a supported model without loading an inference engine."""
    from .hub import download_model

    print(
        download_model(
            args.repo_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_dir=args.local_dir,
            local_files_only=args.local_files_only,
        )
    )


def _run_vad(args):
    """Print frame decisions for a local audio file using the requested engine."""
    import soundfile as sf

    from . import VAD

    with (
        VAD(
            threshold=args.threshold,
            engine=args.engine,
            library_path=args.library_path,
            model_precision=args.model_precision,
            fft_precision=args.fft_precision,
            cache_dir=args.cache_dir,
        ) as detector,
        sf.SoundFile(args.audio) as source,
    ):
        if source.samplerate != 16000 or source.channels != 1:
            raise ValueError("Expected mono 16 kHz audio")
        for block in source.blocks(blocksize=16000, dtype="float32"):
            for frame in detector.push(block):
                print(json.dumps(asdict(frame)))


def _run_minilm(args):
    """Print the classification of one text for models with a head, else its embedding."""
    from . import MiniLM

    options = dict(
        adapter_path=args.adapter,
        threads=args.threads,
        model_precision=args.model_precision,
        chunk_tokens=args.chunk_tokens,
        library_path=args.library_path,
    )
    if args.repo is not None:
        selected = MiniLM.from_pretrained(
            args.repo,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_dir=args.local_dir,
            local_files_only=args.local_files_only,
            **options,
        )
    else:
        selected = MiniLM(args.model, **options)
    with selected as model:
        result = asdict(model.classify(args.text)) if model.head_size else {}
        if args.embedding or not model.head_size:
            result["embedding"] = model.embed(args.text).tolist()
        print(json.dumps(result))


if __name__ == "__main__":
    main()
