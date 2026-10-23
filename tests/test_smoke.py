"""Быстрый smoke-тест: импорты, данные, промпты и формат выходных файлов."""

from src import config, data, prompts


def test_imports() -> None:
    """Проверяет, что основные модули импортируются и конфиг заполнен."""
    assert config.BASE_MODEL
    assert config.MAX_SEQ_LEN > 0
    assert callable(data.load_train_records)
    assert callable(data.sample_few_shot_examples)
    assert callable(prompts.build_messages)


def test_train_datasets() -> None:
    """Проверяет количество, поля и полноту обучающих записей."""
    for style in ("type1", "type2"):
        records = data.load_train_records(style)
        assert len(records) == 25
        for record in records:
            assert {"instruction", "input", "output"} <= record.keys()
            assert record["input"].strip()
            assert record["output"].strip()


def test_sample_few_shot_examples() -> None:
    """Проверяет воспроизводимость и размер случайной few-shot выборки."""
    records = data.load_train_records("type1")
    first = data.sample_few_shot_examples(records)
    second = data.sample_few_shot_examples(records)
    other = data.sample_few_shot_examples(records, seed=config.SEED + 1)

    assert first == second
    assert first != other
    assert len(first) == config.FEW_SHOT_COUNT
    assert len({record["input"] for record in first}) == config.FEW_SHOT_COUNT
    assert all(record in records for record in first)
    assert first != records[: config.FEW_SHOT_COUNT]


def test_test_inputs() -> None:
    """Проверяет, что тестовых входов ровно пять и id уникальны."""
    records = data.load_test_inputs()
    assert len(records) == 5
    assert [record["id"] for record in records] == [1, 2, 3, 4, 5]
    assert all(record["text"].strip() for record in records)


def test_prompts_contain_text_and_style() -> None:
    """Проверяет наличие исходного текста и маркера стиля в промптах."""
    news_text = data.load_test_inputs()[0]["text"]
    for style, marker in prompts.STYLE_MARKERS.items():
        messages = prompts.build_messages(news_text, style)
        system_content = messages[0]["content"]
        user_content = messages[-1]["content"]
        assert marker in system_content
        assert news_text in user_content


def test_outputs_format_if_present() -> None:
    """Проверяет, что готовые outputs содержат ровно пять непустых блоков."""
    for style, path in config.OUTPUT_PATHS.items():
        if not path.exists():
            continue
        blocks = [block.strip() for block in path.read_text(encoding="utf-8").split("---")]
        blocks = [block for block in blocks if block]
        assert len(blocks) == 5, f"{style}: ожидалось 5 блоков, получено {len(blocks)}"
        assert all(len(block) > 50 for block in blocks)
