"""Промпты и chat-шаблоны для двух целевых стилей."""

STYLE_INSTRUCTIONS: dict[str, str] = {
    "type1": "Перепиши новостной текст в стиле Telegram-канала @banki_oil.",
    "type2": (
        "Перепиши новостной текст в стиле Telegram-канала @moscowach "
        "(городской новостной формат с эмодзи в начале, жирным заголовком "
        "и короткими абзацами)."
    ),
}
STYLE_MARKERS: dict[str, str] = {"type1": "@banki_oil", "type2": "@moscowach"}


def build_user_content(news_text: str) -> str:
    """Оборачивает исходный новостной текст в пользовательское сообщение.

    Args:
        news_text: Исходный нейтральный новостной текст.

    Returns:
        Текст сообщения с неизменным исходным текстом.
    """
    return f"Текст:\n{news_text.strip()}"


def build_messages(
    news_text: str,
    style: str,
    examples: list[dict] | None = None,
) -> list[dict]:
    """Собирает chat-сообщения для генерации в заданном стиле.

    Args:
        news_text: Исходный новостной текст.
        style: Идентификатор стиля — ``type1`` или ``type2``.
        examples: Опциональные few-shot примеры с полями ``input`` и ``output``.

    Returns:
        Список сообщений в формате chat-шаблона Qwen.

    Raises:
        KeyError: Если стиль неизвестен.
    """
    messages = [{"role": "system", "content": STYLE_INSTRUCTIONS[style]}]
    for example in examples or []:
        messages.append({"role": "user", "content": build_user_content(example["input"])})
        messages.append({"role": "assistant", "content": example["output"].strip()})
    messages.append({"role": "user", "content": build_user_content(news_text)})
    return messages


def build_training_messages(record: dict) -> list[dict]:
    """Собирает обучающий диалог из записи датасета.

    Args:
        record: Запись с полями ``instruction``, ``input`` и ``output``.

    Returns:
        Список сообщений: system с инструкцией, user с новостью, assistant с ответом.
    """
    return [
        {"role": "system", "content": record["instruction"].strip()},
        {"role": "user", "content": build_user_content(record["input"])},
        {"role": "assistant", "content": record["output"].strip()},
    ]
