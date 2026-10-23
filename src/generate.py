"""Генерация кандидатов в стиле канала с LoRA-адаптером или baseline."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from src import config, data, modeling, prompts


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
    tokenizer = modeling.load_tokenizer()
    model = modeling.load_base_model()
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    device = modeling.get_device()
    examples = (
        data.load_train_records(style)[: config.FEW_SHOT_COUNT]
        if adapter_dir is None
        else None
    )

    results: list[dict] = []
    for record in data.load_test_inputs():
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
        results.append({"id": record["id"], "candidates": candidates})
        print(f"[{style}] id={record['id']}: {len(candidates)} вариантов")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    output_path = (
        args.output or config.ARTIFACTS_DIR / f"candidates_{args.style}.json"
    )
    generate_candidates(args.style, adapter_dir, args.num_candidates, output_path)
    print(f"Результат: {output_path}")


if __name__ == "__main__":
    main()
