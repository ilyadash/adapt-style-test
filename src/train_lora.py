"""Обучение LoRA-адаптеров для стилей type1 и type2."""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments

from src import config, data, modeling, prompts


def encode_example(record: dict, tokenizer) -> dict:
    """Кодирует обучающую запись с маскированием токенов промпта.

    Args:
        record: Запись датасета с полями ``instruction``, ``input`` и ``output``.
        tokenizer: Токенизатор Qwen.

    Returns:
        Словарь с ``input_ids``, ``attention_mask`` и ``labels``.
    """
    messages = prompts.build_training_messages(record)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Токенизация промпта не совпадает с началом полного диалога")
    input_ids = full_ids[: config.MAX_SEQ_LEN]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    labels = labels[: len(input_ids)]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def make_collator(pad_token_id: int):
    """Создаёт collator с правым паддингом для обучения.

    Args:
        pad_token_id: Идентификатор pad-токена.

    Returns:
        Функция, собирающая список примеров в батч.
    """

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Дополняет примеры до максимальной длины в батче."""
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def build_lora_model() -> tuple[PeftModel, PreTrainedTokenizerBase]:
    """Загружает базовую модель и оборачивает её в LoRA.

    Returns:
        Модель с LoRA-адаптером в режиме обучения и её токенизатор.
    """
    tokenizer = modeling.load_tokenizer()
    model = modeling.load_base_model()
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.train()
    peft_model.print_trainable_parameters()
    return peft_model, tokenizer


def encode_records(
    records: list[dict], tokenizer: PreTrainedTokenizerBase
) -> list[dict]:
    """Кодирует все записи датасета для обучения.

    Args:
        records: Записи с полями ``instruction``, ``input`` и ``output``.
        tokenizer: Токенизатор Qwen.

    Returns:
        Список закодированных примеров.
    """
    return [encode_example(record, tokenizer) for record in records]


def make_training_arguments(
    output_dir: Path, epochs: int, max_steps: int | None
) -> TrainingArguments:
    """Создаёт аргументы Trainer с зафиксированными гиперпараметрами.

    Args:
        output_dir: Каталог для чекпоинтов и адаптера.
        epochs: Количество эпох.
        max_steps: Ограничение числа шагов или ``None``.

    Returns:
        Готовые аргументы обучения.
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        optim="adamw_torch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=5,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=config.SEED,
        data_seed=config.SEED,
        use_cache=False,
    )


def save_training_artifacts(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    style: str,
    elapsed_sec: float,
) -> list[dict]:
    """Сохраняет LoRA-адаптер, токенизатор и лог обучения.

    Args:
        trainer: Завершённый Trainer.
        tokenizer: Токенизатор модели.
        output_dir: Каталог для сохранения.
        style: Идентификатор стиля.
        elapsed_sec: Затраченное время в секундах.

    Returns:
        История обучения из логов Trainer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    history = [
        {
            "step": entry.get("step"),
            "epoch": entry.get("epoch"),
            "loss": entry.get("loss"),
            "lr": entry.get("learning_rate"),
        }
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    train_loss = next(
        (
            entry["train_loss"]
            for entry in reversed(trainer.state.log_history)
            if "train_loss" in entry
        ),
        None,
    )
    log = {
        "style": style,
        "steps": trainer.state.global_step,
        "elapsed_sec": elapsed_sec,
        "train_loss": train_loss,
        "history": history,
    }
    (output_dir / "training_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return history


def train_style(
    style: str,
    output_dir: Path,
    epochs: int,
    max_steps: int | None = None,
) -> list[dict]:
    """Обучает LoRA-адаптер для одного стиля через Trainer.

    Args:
        style: Идентификатор стиля — ``type1`` или ``type2``.
        output_dir: Каталог для адаптера и лога обучения.
        epochs: Количество эпох.
        max_steps: Ограничение числа шагов оптимизатора для smoke-запуска.

    Returns:
        История обучения со значениями loss и learning rate.
    """
    modeling.set_seed()
    model, tokenizer = build_lora_model()
    examples = encode_records(data.load_train_records(style), tokenizer)
    train_dataset = Dataset.from_list(examples)
    collator = make_collator(tokenizer.pad_token_id)
    training_args = make_training_arguments(output_dir, epochs, max_steps)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    started = time.time()
    trainer.train()
    elapsed = round(time.time() - started, 1)
    history = save_training_artifacts(
        trainer, tokenizer, output_dir, style, elapsed
    )
    print(f"[{style}] готово: {output_dir}, шагов={trainer.state.global_step}, {elapsed} с")
    return history


def main() -> None:
    """Разбирает аргументы командной строки и запускает обучение."""
    parser = argparse.ArgumentParser(description="Обучение LoRA для стиля")
    parser.add_argument("--style", choices=["type1", "type2"], required=True)
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or config.ADAPTER_DIRS[args.style]
    train_style(args.style, output_dir, args.epochs, args.max_steps)


if __name__ == "__main__":
    main()
