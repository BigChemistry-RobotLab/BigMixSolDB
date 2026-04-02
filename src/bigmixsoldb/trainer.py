from __future__ import annotations

from pathlib import Path
from typing import Any

LLAMA_31_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}
    {% set offset = 1 %}
{% else %}
    {% set offset = 0 %}
{% endif %}

{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {{ '### ' + message['role'] + ':\n'}}
    {% if message['role'] == 'assistant' %}
        {% generation %} {{ message['content'] | trim + eos_token }} {% endgeneration %}
    {% else %}
        {{ message['content'] | trim + eos_token }}
    {% endif %}

{% endfor %}

{% if add_generation_prompt %}
    {{ '### assistant:\n' }}
{% endif %}"""


def normalize_hf_model_name(model_name: str) -> str:
    if "::" not in model_name:
        return model_name

    model_type, normalized_name = model_name.split("::", 1)
    if model_type.lower() not in {"hf", "huggingface"}:
        raise ValueError("Only Hugging Face models are supported by this trainer.")
    return normalized_name


def load_hf_dataset(training_file: str | Path, validation_file: str | Path | None = None):
    from datasets import load_dataset

    training_path = Path(training_file)
    if not training_path.exists():
        raise FileNotFoundError(f"Training file not found: {training_path}")

    data_files: dict[str, str] = {"train": str(training_path)}

    if validation_file is not None:
        validation_path = Path(validation_file)
        if not validation_path.exists():
            raise FileNotFoundError(f"Validation file not found: {validation_path}")
        data_files["validation"] = str(validation_path)

    dataset = load_dataset("json", data_files=data_files)

    try:
        dataset = dataset.remove_columns("doi")
    except ValueError:
        pass

    return dataset


def build_tokenizer(model_name: str, *, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    if model_name == "meta-llama/Llama-3.1-8B-Instruct" and not tokenizer.chat_template:
        tokenizer.chat_template = LLAMA_31_CHAT_TEMPLATE

    return tokenizer


def train_huggingface(
    model_name: str,
    training_file: str | Path,
    validation_file: str | Path | None = None,
    *,
    output_dir: str | Path = "checkpoints",
    **config_kwargs: Any,
) -> Path:
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    normalized_model_name = normalize_hf_model_name(model_name)
    trust_remote_code = bool(config_kwargs.pop("trust_remote_code", False))
    dataset = load_hf_dataset(training_file, validation_file)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation")

    model_reference = normalized_model_name.replace("/", "_")
    checkpoint_path = Path(output_dir) / model_reference
    tokenizer = build_tokenizer(normalized_model_name, trust_remote_code=trust_remote_code)

    do_eval = eval_dataset is not None and bool(config_kwargs.pop("do_eval", True))
    config_kwargs.setdefault("seed", 42)
    config_kwargs.setdefault("logging_steps", 1)
    config_kwargs.setdefault("report_to", "none")
    config_kwargs.setdefault("save_strategy", "no")
    config_kwargs.setdefault("run_name", f"ft_{model_reference}")

    trainer_config = SFTConfig(
        output_dir=str(checkpoint_path),
        do_train=True,
        do_eval=do_eval,
        **config_kwargs,
    )

    trainer = SFTTrainer(
        model=normalized_model_name,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if do_eval else None,
        args=trainer_config,
        processing_class=tokenizer,
    )

    trainer.train()

    if do_eval:
        trainer.evaluate()

    trainer.save_model(str(checkpoint_path))
    tokenizer.save_pretrained(str(checkpoint_path))
    return checkpoint_path