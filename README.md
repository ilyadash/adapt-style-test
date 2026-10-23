# Адаптация Qwen2.5-1.5B-Instruct под стили Telegram-каналов

Проект дообучает русскоязычную LLM переписывать нейтральные новости в стиле двух Telegram-каналов:

- `type1` — аналитика и финансы, по мотивам `@banki_oil`;
- `type2` — городские новости, по мотивам `@moscowach`.

Итоговые тексты: `outputs_type1.txt` и `outputs_type2.txt` (по 5 постов). Архитектурные решения, сравнение «до/после», расчёт памяти для vLLM и ограничения — в `RESULTS.md`.

## Требования

- Python 3.10–3.12 (проверено на 3.12.13);
- [uv](https://docs.astral.sh/uv/) 0.11+;
- NVIDIA GPU с CUDA (проверено на RTX 4070 SUPER, 12 GB);
- для vLLM — Linux или WSL2 (на Windows vLLM нативно не поддерживается).

Проверенные версии: `torch==2.11.0+cu128`, `transformers==5.20.2`, `peft==0.21.2`, `datasets==5.1.0`, `accelerate==1.15.0`.

## Установка

```bash
uv sync --locked
```

Если `uv.lock` недоступен или нужна пересборка окружения:

```bash
uv sync
```

## Данные

Папка `data/` в git не коммитится. Положите туда исходные файлы:

```text
data/train_type1.jsonl   # 25 примеров, поля instruction/input/output
data/train_type2.jsonl   # 25 примеров, те же поля
data/test_inputs.jsonl   # 5 примеров, поля id/text
```

Файлы называются `.jsonl`, хотя в задании указаны как `.json`.

## Проверка

```bash
uv run python -m compileall -q src
uv run pytest tests/test_smoke.py -q
```

Smoke-тест проверяет импорты, количество и поля датасетов, сборку промптов и формат готовых outputs.

## Baseline

Few-shot baseline без дообучения (3 примера на стиль):

```bash
uv run python -m src.baseline --style type1
uv run python -m src.baseline --style type2
```

Результат: `artifacts/baseline_type1.json`, `artifacts/baseline_type2.json`. Эти файлы и адаптеры в git не коммитятся.

## Обучение LoRA

```bash
uv run python -m src.train_lora --style type1
uv run python -m src.train_lora --style type2
```

Быстрая проверка пайплайна:

```bash
uv run python -m src.train_lora --style type1 --max-steps 1 --output-dir artifacts/smoke_lora
```

Адаптеры сохраняются в `artifacts/lora_type1` и `artifacts/lora_type2`, лог обучения — в `training_log.json` рядом с адаптером.

Гиперпараметры зафиксированы в `src/config.py`:

| Параметр | Значение |
|---|---|
| Базовая модель | `Qwen/Qwen2.5-1.5B-Instruct` |
| LoRA rank / alpha / dropout | 8 / 16 / 0.1 |
| Целевые модули | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | 1e-4 |
| Эпохи | 10 |
| Batch / gradient accumulation | 2 / 2 |
| Max sequence length | 1024 |
| Optimizer / scheduler | AdamW / cosine, warmup 10% |
| Seed | 42 |

## Генерация

```bash
uv run python -m src.generate --style type1
uv run python -m src.generate --style type2
```

По умолчанию создаётся 5 кандидатов на каждый тестовый вход:

```text
artifacts/candidates_type1.json
artifacts/candidates_type2.json
```

Параметры генерации: `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.1`, `max_new_tokens=300`.

## Запись outputs

Финальные посты собираются из выбранных кандидатов. Формат selection-файла:

```json
{"1": 0, "2": 1, "3": 0, "4": 2, "5": 0}
```

Ключ — `id` тестового входа, значение — индекс кандидата. Без `--selection` берётся первый кандидат.

```bash
uv run python -m src.write_outputs \
  --style type1 \
  --candidates artifacts/candidates_type1.json \
  --selection artifacts/selection_type1.json
```

Формат outputs: 5 постов в порядке `id`, разделитель — строка `---`.

## Воспроизведение baseline и LoRA

- Seed фиксирован (`42`), но генерация на GPU может незначительно отличаться из-за недетерминизма CUDA.
- Финальные outputs выбраны вручную из пяти кандидатов на каждый вход; selection-файлы находятся в `artifacts/` и в git не входят.
- Для точного повторения финальных файлов используйте уже готовые `outputs_type1.txt` и `outputs_type2.txt`; пайплайн воспроизводит формат и стиль, но не гарантирует побайтовое совпадение кандидатов.

## Структура проекта

```text
src/
  config.py        # пути, модель, seed, гиперпараметры
  data.py          # загрузка и валидация JSONL
  prompts.py       # chat-шаблоны двух стилей
  modeling.py      # загрузка токенизатора и модели
  baseline.py      # few-shot baseline
  train_lora.py    # обучение LoRA через Trainer
  generate.py      # генерация кандидатов
  write_outputs.py # выбор кандидатов и запись outputs
tests/test_smoke.py
outputs_type1.txt
outputs_type2.txt
RESULTS.md
pyproject.toml
uv.lock
```

## vLLM

Краткая команда запуска и расчёт памяти для 24 GB VRAM и 15 одновременных запросов — в `RESULTS.md`, раздел «vLLM и память».

## Известные ограничения

- Обучение на 25 примерах на стиль: модель уверенно копирует структуру канала, но может добавлять факты, которых нет во входном тексте.
- Оценка стиля ручная; BLEU/ROUGE не использовались.
- Лучшие кандидаты выбирались вручную, поэтому финальные outputs не являются полностью автоматическими.
- vLLM не запускался локально: на Windows он не поддерживается без WSL2.
- Адаптеры и кандидаты не коммитятся; их нужно воспроизвести командами выше.
