"""Запись финальных outputs из выбранных кандидатов."""

import argparse
import json
from pathlib import Path

from src import config


def load_candidates(candidates_path: Path) -> list[dict]:
    """Загружает кандидатов из JSON.

    Args:
        candidates_path: Путь к JSON с кандидатами.

    Returns:
        Список записей с ``id`` и ``candidates``.
    """
    return json.loads(candidates_path.read_text(encoding="utf-8"))


def load_selection(selection_path: Path | None) -> dict:
    """Загружает карту выбранных индексов или пустую карту.

    Args:
        selection_path: Путь к JSON с выбором или ``None``.

    Returns:
        Словарь вида ``{"1": 0, ...}``.
    """
    if selection_path is None:
        return {}
    return json.loads(selection_path.read_text(encoding="utf-8"))


def validate_candidates(records: list[dict]) -> list[dict]:
    """Проверяет наличие кандидатов для id=1..5 и сортирует их.

    Args:
        records: Записи с кандидатами.

    Returns:
        Записи, отсортированные по ``id``.

    Raises:
        ValueError: Если набор id не совпадает с 1..5.
    """
    records = sorted(records, key=lambda item: item["id"])
    if [record["id"] for record in records] != [1, 2, 3, 4, 5]:
        raise ValueError("Ожидались кандидаты для id=1..5")
    return records


def select_post(record: dict, selection: dict) -> str:
    """Выбирает и проверяет один пост по индексу из selection.

    Args:
        record: Запись с ``id`` и списком ``candidates``.
        selection: Карта выбранных индексов.

    Returns:
        Непустой текст выбранного кандидата.

    Raises:
        ValueError: Если индекс вне диапазона или кандидат пуст.
    """
    index = int(selection.get(str(record["id"]), 0))
    candidates = record["candidates"]
    if not 0 <= index < len(candidates):
        raise ValueError(f"id={record['id']}: индекс {index} вне диапазона")
    post = candidates[index].strip()
    if not post:
        raise ValueError(f"id={record['id']}: пустой кандидат")
    return post


def collect_posts(records: list[dict], selection: dict) -> list[str]:
    """Собирает посты в порядке ``id``.

    Args:
        records: Отсортированные записи с кандидатами.
        selection: Карта выбранных индексов.

    Returns:
        Список текстов постов.
    """
    return [select_post(record, selection) for record in records]


def write_outputs(
    candidates_path: Path,
    selection_path: Path | None,
    output_path: Path,
) -> None:
    """Собирает пять постов в порядке ``id`` и сохраняет через разделитель ``---``.

    Args:
        candidates_path: JSON с кандидатами по каждому тестовому входу.
        selection_path: JSON вида ``{"1": 0, ...}`` с индексом выбранного варианта.
        output_path: Путь итогового txt-файла.

    Raises:
        ValueError: Если кандидатов не пять, id не совпадают или индекс вне диапазона.
    """
    records = validate_candidates(load_candidates(candidates_path))
    selection = load_selection(selection_path)
    posts = collect_posts(records, selection)
    output_path.write_text("\n\n---\n\n".join(posts) + "\n", encoding="utf-8")
    print(f"Записано {len(posts)} постов: {output_path}")


def main() -> None:
    """Разбирает аргументы командной строки и записывает outputs."""
    parser = argparse.ArgumentParser(description="Запись финальных outputs")
    parser.add_argument("--style", choices=["type1", "type2"], required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    write_outputs(
        args.candidates,
        args.selection,
        args.output or config.OUTPUT_PATHS[args.style],
    )


if __name__ == "__main__":
    main()
