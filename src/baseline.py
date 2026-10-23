"""Few-shot baseline: генерация без дообучения для сравнения «до/после»."""

import argparse
import json
from pathlib import Path

import torch

from src import config, data, modeling, prompts


def generate_baseline(style: str, num_candidates: int, output_path: Path) -> list[dict]:
    """Генерирует ответы базовой моделью с few-shot примерами.

    Args:
        style: Идентификатор стиля — ``type1`` или ``type2``.
        num_candidates: Количество вариантов на каждый тестовый вход.
        output_path: Путь для JSON с кандидатами.

    Returns:
        Список словарей с ``id`` и списком ``candidates``.
    """
    modeling.set_seed()
    tokenizer = modeling.load_tokenizer()
    model = modeling.load_base_model()
    device = modeling.get_device()
    examples = data.load_train_records(style)[: config.FEW_SHOT_COUNT]

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
        print(f"[{style}] id={record['id']}: {len(candidates[0])} символов")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    """Разбирает аргументы командной строки и запускает baseline-генерацию."""
    parser = argparse.ArgumentParser(description="Few-shot baseline без LoRA")
    parser.add_argument("--style", choices=["type1", "type2"], required=True)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or config.ARTIFACTS_DIR / f"baseline_{args.style}.json"
    generate_baseline(args.style, args.num_candidates, output_path)
    print(f"Результат: {output_path}")


if __name__ == "__main__":
    main()
