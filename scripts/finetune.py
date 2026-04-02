from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from bigmixsoldb.trainer import train_huggingface


def load_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded_config = yaml.safe_load(handle) or {}

    if not isinstance(loaded_config, dict):
        raise ValueError("Finetuning config must be a mapping.")

    return dict(loaded_config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Hugging Face chat model on BigMixSolDB extraction data."
    )
    parser.add_argument(
        "--model",
        help="Hugging Face model name. Optional if provided in --config.",
    )
    parser.add_argument(
        "--config",
        help="YAML or JSON config file with SFTConfig fields and an optional model entry.",
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="Training dataset in JSONL format.",
    )
    parser.add_argument(
        "--validation-data",
        "--val-data",
        dest="validation_data",
        help="Optional validation dataset in JSONL format.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where the trained checkpoint will be saved.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    model_type = str(config.pop("model_type", "hf")).lower()
    if model_type not in {"hf", "huggingface"}:
        raise ValueError("Only Hugging Face finetuning is supported by this script.")

    model_name = args.model or config.pop("model", None)
    if not model_name:
        raise ValueError("Provide --model or define model in the config file.")

    config_output_dir = config.pop("output_dir", config.pop("checkpoint_dir", "checkpoints"))
    output_dir = args.output_dir or str(config_output_dir)

    checkpoint_path = train_huggingface(
        model_name=model_name,
        training_file=args.train_data,
        validation_file=args.validation_data,
        output_dir=output_dir,
        **config,
    )
    print(f"Saved Hugging Face checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()