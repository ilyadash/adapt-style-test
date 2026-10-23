"""Генерация кандидатов в стиле канала с LoRA-адаптером или baseline."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import PreTrainedTokenizerBase

from src import config, data, modeling, prompts


def load_generation_model(
    adapter_dir: Path | None,
) -> tuple[PreTrainedTokenizerBase, torch.nn.Module, torch.device]:
    """Загружает токенизатор, базовую модель и LoRA-адаптер.

    Args:
        adapter_dir: Каталог LoRA-адаптера или ``None`` для baseline.

    Returns:
        Токенизатор, модель в режиме оценки и устройство инференса.
    """
    tokenizer = modeling.load_tokenizer()
    model = modeling.load_base_model()
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model, modeling.get_device()


def load_few_shot_examples(
    style: str, adapter_dir: Path | None
) -> list[dict] | None:
    """Возвращает few-shot примеры для baseline или ``None`` для LoRA.

    Args:
        style: Идентификатор стиля — ``type1`` или ``type2``.
        adapter_dir: Каталог LoRA-адаптера или ``None`` для baseline.

    Returns:
        Список примеров для few-shot промпта или ``None``.
    """
    if adapter_dir is not None:
        return None
    return data.load_train_records(style)[: config.FEW_SHOT_COUNT]


def generate_for_record(
    record: dict,
    style: str,
    examples: list[dict] | None,
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    device: torch.device,
    num_candidates: int,
) -> dict:
    """Генерирует варианты ответа для одного тестового входа.

    Args:
        record: Тестовый вход с полями ``id`` и ``text``.
        style: Идентификатор стиля — ``type1`` или ``type2``.
        examples: Few-shot примеры или ``None``.
        tokenizer: Токенизатор модели.
        model: Модель или LoRA-адаптер.
        device: Устройство инференса.
        num_candidates: Количество вариантов.

    Returns:
        Словарь с ``id`` и списком ``candidates``.
    """
    messages = prompts.build_messages(record["text"], style, examples)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=config.GEN_TEMPERATURE,
            top_p=config.GEN_TOP_P,
            repetition_penalty=config.GEN_REPETITION_PENALTY,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS,
            num_return_sequences=num_candidates,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    candidates = [
        tokenizer.decode(row, skip_special_tokens=True).strip() for row in new_tokens
    ]
    print(f"[{style}] id={record['id']}: {len(candidates)} вариантов")
    return {"id": record["id"], "candidates": candidates}


def save_candidates(results: list[dict], output_path: Path) -> None:
    """Сохраняет сгенерированные кандидаты в JSON.

    Args:
        results: Результаты генерации по каждому входу.
        output_path: Путь для JSON-файла.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def generate_candidates(
    style: str,
    adapter_dir: Path | None,
    num_candidates: int,
    output_path: Path,
) -> list[dict]:
    """Генерирует варианты ответов для всех тестовых входов.

    Args:
        style: Идентификатор стиля — ``type1`` или ``type2``.
        adapter_dir: Каталог LoRA-адаптера; ``None`` включает few-shot baseline.
        num_candidates: Количество вариантов на каждый вход.
        output_path: Путь для JSON с кандидатами.

    Returns:
        Список словарей с ``id`` и списком ``candidates``.
    """
    modeling.set_seed()
    tokenizer, model, device = load_generation_model(adapter_dir)
    examples = load_few_shot_examples(style, adapter_dir)
    results = [
        generate_for_record(
            record, style, examples, tokenizer, model, device, num_candidates
        )
        for record in data.load_test_inputs()
    ]
    save_candidates(results, output_path)
    return results


def main() -> None:
    """Разбирает аргументы командной строки и запускает генерацию."""
    parser = argparse.ArgumentParser(description="Генерация текстов в стиле канала")
    parser.add_argument("--style", choices=["type1", "type2"], required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--num-candidates", type=int, default=config.GEN_NUM_CANDIDATES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    adapter_dir = None if args.baseline else args.adapter or config.ADAPTER_DIRS[args.style]
    if adapter_dir is not None and not adapter_dir.exists():
        raise FileNotFoundError(f"Адаптер не найден: {adapter_dir}")
    output_path = args.output or config.ARTIFACTS_DIR / f"candidates_{args.style}.json"
    generate_candidates(args.style, adapter_dir, args.num_candidates, output_path)
    print(f"Результат: {output_path}")


if __name__ == "__main__":
    main()
