"""Общие функции загрузки токенизатора и базовой модели."""

import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config


def set_seed(seed: int = config.SEED) -> None:
    """Фиксирует seed для воспроизводимой генерации.

    Args:
        seed: Значение seed для ``random`` и PyTorch.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Возвращает CUDA-устройство, если оно доступно, иначе CPU.

    Returns:
        Устройство для загрузки модели.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_name: str = config.BASE_MODEL) -> AutoTokenizer:
    """Загружает токенизатор и назначает pad-токен.

    Args:
        model_name: Идентификатор модели на Hugging Face.

    Returns:
        Токенизатор с заполненным ``pad_token``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str = config.BASE_MODEL) -> AutoModelForCausalLM:
    """Загружает базовую модель в bfloat16 на GPU или float32 на CPU.

    Args:
        model_name: Идентификатор модели на Hugging Face.

    Returns:
        Модель в режиме оценки.
    """
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model
