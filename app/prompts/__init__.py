"""Завантаження системних промптів із файлів.

Промпти тримаються у markdown-файлах, а не в інлайн-рядках: так їх зручно
версіонувати й переглядати діфи окремо від коду.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=8)
def load_prompt(name: str) -> str:
    """Прочитати текст промпту за іменем файлу без розширення."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Промпт '{name}' не знайдено за шляхом {path}")
    return path.read_text(encoding="utf-8").strip()
