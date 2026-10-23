"""Загрузка и валидация JSONL-датасетов проекта."""

import json
import random
from pathlib import Path

from src import config


def load_jsonl(path: Path) -> list[dict]:
    """Загружает JSONL-файл и проверяет, что он не пустой.

    Args:
        path: Путь к JSONL-файлу.

    Returns:
        Список записей из файла.

    Raises:
        FileNotFoundError: Если файл не существует.
        ValueError: Если файл пуст или содержит некорректный JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: некорректный JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: ожидался JSON-объект")
            records.append(record)
    if not records:
        raise ValueError(f"Файл пуст: {path}")
    return records


def load_train_records(style: str) -> list[dict]:
    """Загружает обучающие записи для указанного стиля.

    Args:
        style: Идентификатор стиля — ``type1`` или ``type2``.

    Returns:
        Список записей с непустыми полями ``instruction``, ``input`` и ``output``.

    Raises:
        KeyError: Если стиль неизвестен.
        ValueError: Если в записи отсутствует обязательное поле.
    """
    if style not in config.TRAIN_PATHS:
        raise KeyError(f"Неизвестный стиль: {style}")
    records = load_jsonl(config.TRAIN_PATHS[style])
    for index, record in enumerate(records, start=1):
        for field in ("instruction", "input", "output"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{style}, запись {index}: пустое поле {field}")
    return records


def split_train_validation(
    records: list[dict],
    val_size: int = config.VAL_SIZE,
    seed: int = config.SEED,
) -> tuple[list[dict], list[dict]]:
    """Делит записи на train и validation с фиксированным seed.

    Args:
        records: Полный список обучающих записей.
        val_size: Размер validation-выборки.
        seed: Seed для детерминированного перемешивания.

    Returns:
        Кортеж ``(train_records, val_records)`` без пересечений.

    Raises:
        ValueError: Если ``val_size`` не помещается в выборку.
    """
    if not 0 < val_size < len(records):
        raise ValueError(
            f"val_size={val_size} должен быть больше 0 и меньше {len(records)}"
        )
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:val_size])
    train_records = [
        record for index, record in enumerate(records) if index not in val_indices
    ]
    val_records = [records[index] for index in indices[:val_size]]
    return train_records, val_records


def load_test_inputs() -> list[dict]:
    """Загружает тестовые входы в порядке возрастания ``id``.

    Returns:
        Список записей с полями ``id`` (int) и ``text`` (str).

    Raises:
        ValueError: Если ``id`` или ``text`` отсутствуют либо ``id`` повторяются.
    """
    records = load_jsonl(config.TEST_INPUTS_PATH)
    seen_ids: set[int] = set()
    for index, record in enumerate(records, start=1):
        record_id = record.get("id")
        text = record.get("text")
        if not isinstance(record_id, int):
            raise ValueError(f"Тестовый вход {index}: поле id должно быть целым числом")
        if record_id in seen_ids:
            raise ValueError(f"Тестовый вход {index}: повторяющийся id={record_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Тестовый вход {index}: пустое поле text")
        seen_ids.add(record_id)
    return sorted(records, key=lambda item: item["id"])
