#!/usr/bin/env python
"""Запуск тестов с покрытием"""

import pytest
import sys

if __name__ == "__main__":
    # Запускаем тесты с покрытием
    args = [
        "tests/",
        "-v",  # Подробный вывод
        "--cov=.",  # Измерять покрытие для всего проекта
        "--cov-report=term-missing",  # Показать непокрытые строки
        "--cov-report=html",  # Создать HTML отчет
        "--cov-fail-under=60",  # Требовать минимум 60% покрытия
    ]

    # Добавляем опции для пропуска медленных тестов
    if "--skip-slow" in sys.argv:
        args.append("-m 'not slow'")

    sys.exit(pytest.main(args))