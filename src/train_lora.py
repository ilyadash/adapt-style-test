"""Обучение LoRA-адаптеров для стилей type1 и type2."""

import argparse
import contextlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedTokenizerBase, get_cosine_schedule_with_warmup

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


def count_optimizer_steps(
    num_examples: int, epochs: int, max_steps: int | None
) -> int:
    """Считает число шагов оптимизатора с учётом накопления градиента.

    Args:
        num_examples: Количество обучающих примеров.
        epochs: Количество эпох.
        max_steps: Ограничение числа шагов для smoke-запуска или ``None``.

    Returns:
        Итоговое число шагов оптимизатора.
    """
    micro_steps_per_epoch = math.ceil(num_examples / config.TRAIN_BATCH_SIZE)
    steps_per_epoch = math.ceil(
        micro_steps_per_epoch / config.GRADIENT_ACCUMULATION_STEPS
    )
    total_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    return total_steps


def build_optimizer_and_scheduler(
    model: PeftModel, total_steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Создаёт AdamW и cosine-планировщик для обучения LoRA.

    Args:
        model: Модель с LoRA-адаптером.
        total_steps: Итоговое число шагов оптимизатора.

    Returns:
        Оптимизатор и планировщик learning rate.
    """
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    warmup_steps = max(1, int(total_steps * config.WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler


def build_autocast(device: torch.device) -> contextlib.AbstractContextManager:
    """Возвращает bf16 autocast на CUDA или пустой контекст на CPU.

    Args:
        device: Устройство обучения.

    Returns:
        Контекстный менеджер для прямого прохода.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_training_loop(
    model: PeftModel,
    examples: list[dict],
    collator: Callable[[list[dict]], dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    style: str,
    epochs: int,
    total_steps: int,
    max_steps: int | None,
) -> list[dict]:
    """Выполняет цикл обучения по эпохам и микробатчам.

    Args:
        model: Модель с LoRA-адаптером в режиме обучения.
        examples: Закодированные обучающие примеры.
        collator: Функция сборки батча.
        optimizer: Оптимизатор.
        scheduler: Планировщик learning rate.
        device: Устройство обучения.
        style: Идентификатор стиля для логов.
        epochs: Количество эпох.
        total_steps: Итоговое число шагов оптимизатора.
        max_steps: Ограничение числа шагов или ``None``.

    Returns:
        История обучения со значениями loss и learning rate.
    """
    batch_size = config.TRAIN_BATCH_SIZE
    accumulation = config.GRADIENT_ACCUMULATION_STEPS
    autocast = build_autocast(device)
    history: list[dict] = []
    global_step = 0

    for epoch in range(epochs):
        order = torch.randperm(len(examples)).tolist()
        for batch_start in range(0, len(examples), batch_size):
            if max_steps is not None and global_step >= max_steps:
                break
            micro_step = batch_start // batch_size + 1
            is_last_micro_step = batch_start + batch_size >= len(examples)
            batch_indices = order[batch_start : batch_start + batch_size]
            batch = collator([examples[index] for index in batch_indices])
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast:
                loss = model(**batch).loss / accumulation
            loss.backward()
            if micro_step % accumulation == 0 or is_last_micro_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                history.append(
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": round(float(loss.item()) * accumulation, 4),
                        "lr": round(scheduler.get_last_lr()[0], 8),
                    }
                )
                if global_step % 5 == 0 or global_step == total_steps:
                    print(
                        f"[{style}] step {global_step}/{total_steps} "
                        f"loss={history[-1]['loss']:.4f}"
                    )
        if max_steps is not None and global_step >= max_steps:
            break

    return history


def save_training_artifacts(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    style: str,
    history: list[dict],
    steps: int,
    elapsed_sec: float,
) -> None:
    """Сохраняет LoRA-адаптер, токенизатор и лог обучения.

    Args:
        model: Обученная модель с LoRA-адаптером.
        tokenizer: Токенизатор модели.
        output_dir: Каталог для сохранения.
        style: Идентификатор стиля.
        history: История обучения.
        steps: Число выполненных шагов оптимизатора.
        elapsed_sec: Затраченное время в секундах.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log = {
        "style": style,
        "steps": steps,
        "elapsed_sec": elapsed_sec,
        "history": history,
    }
    (output_dir / "training_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train_style(
    style: str,
    output_dir: Path,
    epochs: int,
    max_steps: int | None = None,
) -> list[dict]:
    """Обучает LoRA-адаптер для одного стиля.

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
    collator = make_collator(tokenizer.pad_token_id)
    device = modeling.get_device()
    total_steps = count_optimizer_steps(len(examples), epochs, max_steps)
    optimizer, scheduler = build_optimizer_and_scheduler(model, total_steps)

    started = time.time()
    history = run_training_loop(
        model=model,
        examples=examples,
        collator=collator,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        style=style,
        epochs=epochs,
        total_steps=total_steps,
        max_steps=max_steps,
    )
    elapsed = round(time.time() - started, 1)
    save_training_artifacts(
        model,
        tokenizer,
        output_dir,
        style,
        history,
        steps=len(history),
        elapsed_sec=elapsed,
    )
    print(f"[{style}] готово: {output_dir}, шагов={len(history)}, {elapsed} с")
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
