"""
Модуль для решения матричных игр и поиска равновесия Нэша
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import nashpy as nash
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ========== Pydantic схемы данных ==========

class MarketData(BaseModel):
    """
    Рыночные данные - источник информации о рынке
    Все расчеты ведутся на основе этих данных
    """
    costs_a: List[float] = Field(..., description="Себестоимость для стратегий игрока A", min_items=2, max_items=2)
    costs_b: List[float] = Field(..., description="Себестоимость для стратегий игрока B", min_items=2, max_items=2)
    demand_matrix: List[List[float]] = Field(..., description="Спрос Q(i,j) для каждой пары стратегий", min_items=2,
                                             max_items=2)

    @validator('demand_matrix')
    def validate_demand_matrix(cls, v):
        if len(v) != 2 or len(v[0]) != 2 or len(v[1]) != 2:
            raise ValueError("demand_matrix должна быть размером 2x2")
        return v

    @validator('costs_a', 'costs_b', pre=True)
    def validate_costs(cls, v):
        """Валидация: себестоимость не может быть отрицательной"""
        if any(cost < 0 for cost in v):
            raise ValueError('Себестоимость не может быть отрицательной')
        return v

    def calculate_profit_matrices(self, prices_a: List[float], prices_b: List[float]) -> Tuple[
        List[List[float]], List[List[float]]]:
        """
        Рассчитать матрицы прибыли на основе цен и спроса
        Прибыль = (Цена - Себестоимость) × Спрос
        """
        profit_a = [[0, 0], [0, 0]]
        profit_b = [[0, 0], [0, 0]]

        for i in range(2):
            for j in range(2):
                profit_a[i][j] = (prices_a[i] - self.costs_a[i]) * self.demand_matrix[i][j]
                profit_b[i][j] = (prices_b[j] - self.costs_b[j]) * self.demand_matrix[i][j]

        return profit_a, profit_b


class EquilibriumResult(BaseModel):
    """Результат расчета равновесия Нэша для двух продавцов"""

    probs_a: List[float] = Field(
        description="Вероятности выбора стратегий игроком A [p1, p2]",
        example=[0.5, 0.5]
    )

    probs_b: List[float] = Field(
        description="Вероятности выбора стратегий игроком B [q1, q2]",
        example=[0.7, 0.3]
    )

    equilibrium_type: str = Field(
        description="Тип равновесия: 'mixed' (смешанное), 'pure_1' (чистая стратегия 1), 'pure_2' (чистая стратегия 2)",
        example="mixed"
    )

    expected_profit_a: float = Field(
        description="Ожидаемая прибыль игрока A в равновесии (руб.)",
        example=1250.50
    )

    expected_profit_b: float = Field(
        description="Ожидаемая прибыль игрока B в равновесии (руб.)",
        example=1100.25
    )


# ========== Хранилище состояния игры ==========

class GameState:
    """
    Синглтон для хранения рыночных данных
    Хранит все данные игры между запросами
    """
    _instance = None
    market_data: Optional[MarketData] = None
    _equilibrium_cache: Optional[EquilibriumResult] = None
    _reference_prices_a: Optional[List[float]] = None
    _reference_prices_b: Optional[List[float]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_market_data(self, market_data: MarketData, reference_prices_a: List[float],
                        reference_prices_b: List[float]):
        """
        Загрузить рыночные данные
        """
        self.market_data = market_data
        self._reference_prices_a = reference_prices_a
        self._reference_prices_b = reference_prices_b
        self._equilibrium_cache = None  # Сбрасываем кэш
        logger.info(f"Рыночные данные загружены")
        logger.info(f"  Себестоимость A: {market_data.costs_a}")
        logger.info(f"  Себестоимость B: {market_data.costs_b}")
        logger.info(f"  Матрица спроса: {market_data.demand_matrix}")
        logger.info(f"  Референсные цены A: {reference_prices_a}")
        logger.info(f"  Референсные цены B: {reference_prices_b}")

    def get_equilibrium(self) -> EquilibriumResult:
        """Получить равновесие Нэша (с кэшированием)"""
        if self._equilibrium_cache is None:
            self._equilibrium_cache = self._calculate_equilibrium()
        return self._equilibrium_cache

    def _calculate_equilibrium(self) -> EquilibriumResult:
        """Рассчитать равновесие Нэша на основе рыночных данных"""
        if self.market_data is None:
            raise ValueError("Рыночные данные не загружены")

        # Рассчитываем матрицы прибыли при референсных ценах
        profit_a, profit_b = self.market_data.calculate_profit_matrices(
            self._reference_prices_a,
            self._reference_prices_b
        )

        # Находим равновесие
        game = nash.Game(np.array(profit_a), np.array(profit_b))
        equilibria = list(game.support_enumeration())

        if not equilibria:
            raise ValueError("Равновесий не найдено")

        eq_a, eq_b = equilibria[0]

        # Определяем тип равновесия
        if eq_a[0] > 1e-6 and eq_a[1] > 1e-6:
            eq_type = "mixed"
        elif eq_a[0] > 1e-6:
            eq_type = "pure_1"
        else:
            eq_type = "pure_2"

        # Рассчитываем ожидаемую прибыль
        expected_profit_a = sum(profit_a[0][j] * eq_b[j] for j in range(2)) * eq_a[0] + \
                            sum(profit_a[1][j] * eq_b[j] for j in range(2)) * eq_a[1]
        expected_profit_b = sum(profit_b[i][0] * eq_a[i] for i in range(2)) * eq_b[0] + \
                            sum(profit_b[i][1] * eq_a[i] for i in range(2)) * eq_b[1]

        logger.info(f"Равновесие рассчитано: тип={eq_type}, p={eq_a}, q={eq_b}")
        logger.info(f"Ожидаемая прибыль: A={expected_profit_a:.2f}, B={expected_profit_b:.2f}")

        return EquilibriumResult(
            probs_a=list(eq_a),
            probs_b=list(eq_b),
            equilibrium_type=eq_type,
            expected_profit_a=expected_profit_a,
            expected_profit_b=expected_profit_b
        )

    def get_market_data(self):
        if self.market_data is None:
            from fastapi import HTTPException
            raise HTTPException(400, "Рыночные данные не загружены. Сначала POST /game/set_market_data")
        return self.market_data

    def get_reference_prices_a(self) -> List[float]:
        """Получить референсные цены игрока A"""
        if self._reference_prices_a is None:
            raise ValueError("Референсные цены A не загружены")
        return self._reference_prices_a

    def get_reference_prices_b(self) -> List[float]:
        """Получить референсные цены игрока B"""
        if self._reference_prices_b is None:
            raise ValueError("Референсные цены B не загружены")
        return self._reference_prices_b