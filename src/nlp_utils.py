# nlp_utils.py
import re
from collections import Counter


def tokenize(text: str):
    """Разбивает текст на слова, приводя к нижнему регистру"""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def word_frequencies(text: str):
    """Возвращает частотный словарь слов"""
    tokens = tokenize(text)
    return Counter(tokens)
