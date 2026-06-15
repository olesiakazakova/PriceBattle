# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from nash_equilibrium import GameState, MarketData
from lp_optimizer import optimize_margin_logic


@pytest.fixture
def client():
    """Тестовый клиент FastAPI"""
    return TestClient(app)


@pytest.fixture
def sample_market_data():
    """Стандартные рыночные данные для тестов"""
    return {
        "costs_a": [200.0, 350.0],
        "costs_b": [100.0, 300.0],
        "demand_matrix": [[150.0, 70.0], [150.0, 70.0]],
        "reference_prices_a": [300.0, 400.0],
        "reference_prices_b": [250.0, 400.0]
    }


@pytest.fixture
def symmetric_market_data():
    """Симметричные рыночные данные"""
    return {
        "costs_a": [200.0, 200.0],
        "costs_b": [200.0, 200.0],
        "demand_matrix": [[100.0, 80.0], [80.0, 100.0]],
        "reference_prices_a": [300.0, 300.0],
        "reference_prices_b": [300.0, 300.0]
    }


@pytest.fixture
def dominant_strategy_data():
    """Данные с доминирующей стратегией"""
    return {
        "costs_a": [100.0, 500.0],
        "costs_b": [100.0, 500.0],
        "demand_matrix": [[200.0, 50.0], [180.0, 30.0]],
        "reference_prices_a": [250.0, 600.0],
        "reference_prices_b": [250.0, 600.0]
    }


@pytest.fixture
def cleanup_state():
    """Очистка состояния после тестов"""
    yield
    GameState._instance = None
    GameState.market_data = None
    GameState._equilibrium_cache = None