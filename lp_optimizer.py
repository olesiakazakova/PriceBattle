"""
Модуль для оптимизации маржи с использованием линейного программирования
"""

import logging
from typing import List

from pydantic import BaseModel, Field, validator
from pulp import LpProblem, LpMaximize, LpVariable, LpStatus, value, PULP_CBC_CMD

logger = logging.getLogger(__name__)


# ========== Pydantic схемы данных ==========

class MarginOptimizationRequest(BaseModel):
    """Запрос на оптимизацию минимальной маржи"""
    min_margin: float = Field(..., description="Минимальная маржа (руб.)", ge=0)
    max_price: float = Field(200, description="Максимальная допустимая цена (руб.)", ge=0)


# ========== Логика оптимизации ==========

def optimize_margin_logic(
        market_data,
        equilibrium,
        reference_prices_a: List[float],
        reference_prices_b: List[float],
        min_margin: float,
        max_price: float
):
    """
    Выполняет оптимизацию минимальной маржи через линейное программирование

    Args:
        market_data: Объект MarketData с рыночными данными
        equilibrium: Объект EquilibriumResult с равновесием Нэша
        reference_prices_a: Референсные цены игрока A
        reference_prices_b: Референсные цены игрока B
        min_margin: Минимальная требуемая маржа
        max_price: Максимальная допустимая цена

    Returns:
        dict: Результаты оптимизации
    """

    logger.info(f"Рыночные данные:")
    logger.info(f"  Себестоимость A: {market_data.costs_a}")
    logger.info(f"  Себестоимость B: {market_data.costs_b}")
    logger.info(f"  Матрица спроса: {market_data.demand_matrix}")

    logger.info(f"Равновесие:")
    logger.info(f"  Тип: {equilibrium.equilibrium_type}")
    logger.info(f"  Вероятности A: {equilibrium.probs_a}")
    logger.info(f"  Вероятности B: {equilibrium.probs_b}")

    # ========== РАСЧЕТ ОЖИДАЕМОГО СПРОСА ==========
    q_eq = equilibrium.probs_b

    expected_demand_1 = sum(market_data.demand_matrix[0][j] * q_eq[j] for j in range(2))
    expected_demand_2 = sum(market_data.demand_matrix[1][j] * q_eq[j] for j in range(2))

    logger.info(f"Ожидаемый спрос:")
    logger.info(f"  Стратегия 1: {expected_demand_1:.2f}")
    logger.info(f"  Стратегия 2: {expected_demand_2:.2f}")

    # ========== ПОСТРОЕНИЕ ЗАДАЧИ ЛП ==========
    prob = LpProblem("Margin_Optimization", LpMaximize)

    # Переменные: маржи для стратегий A
    m1 = LpVariable("margin_1", lowBound=0)
    m2 = LpVariable("margin_2", lowBound=0)
    t = LpVariable("t", lowBound=0)  # минимальная маржа

    # Целевая функция: максимизировать минимальную маржу
    prob += t, "Maximize_Min_Margin"

    # Пользовательские ограничения маржи
    prob += t >= min_margin

    # Базовые ограничения
    prob += m1 >= t
    prob += m2 >= t

    # Ограничения на цены
    prob += market_data.costs_a[0] + m1 <= max_price
    prob += market_data.costs_a[1] + m2 <= max_price

    # Максимальное изменение цены относительно равновесной
    current_price_1 = (reference_prices_a[0] * equilibrium.probs_a[0] +
                       reference_prices_a[1] * equilibrium.probs_a[1])
    current_price_2 = (reference_prices_b[0] * equilibrium.probs_b[0] +
                       reference_prices_b[1] * equilibrium.probs_b[1])

    max_price_change_percent = 0.3  # не более 30%
    prob += m1 + market_data.costs_a[0] <= current_price_1 * (1 + max_price_change_percent)
    prob += m1 + market_data.costs_a[0] >= current_price_1 * (1 - max_price_change_percent)
    prob += m2 + market_data.costs_a[1] <= current_price_2 * (1 + max_price_change_percent)
    prob += m2 + market_data.costs_a[1] >= current_price_2 * (1 - max_price_change_percent)

    # ========== РЕШЕНИЕ ЗАДАЧИ ==========
    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    logger.info(f"Статус решения: {LpStatus[prob.status]}")

    if prob.status != 1:
        return {
            "status": "error",
            "message": f"Не удалось найти оптимальное решение. Статус: {LpStatus[prob.status]}",
            "suggestion": "Попробуйте увеличить max_price"
        }

    # ========== ФОРМИРОВАНИЕ РЕЗУЛЬТАТА ==========
    optimal_t = value(t)
    optimal_m1 = value(m1)
    optimal_m2 = value(m2)

    # Расчет прибыли
    profit_1 = optimal_m1 * expected_demand_1
    profit_2 = optimal_m2 * expected_demand_2

    # Определяем активные ограничения
    active_constraints = []
    if abs(optimal_m1 - optimal_t) < 0.01:
        active_constraints.append("m1 = t")
    if abs(optimal_m2 - optimal_t) < 0.01:
        active_constraints.append("m2 = t")
    if abs(market_data.costs_a[0] + optimal_m1 - max_price) < 0.01:
        active_constraints.append(f"цена1 = {max_price}")
    if abs(market_data.costs_a[1] + optimal_m2 - max_price) < 0.01:
        active_constraints.append(f"цена2 = {max_price}")

    result = {
        "status": "success",
        "optimization_status": LpStatus[prob.status],

        # Основные результаты
        "optimal_margins": {
            "strategy_1": round(optimal_m1, 2),
            "strategy_2": round(optimal_m2, 2),
        },

        "optimal_prices": {
            "strategy_1": round(market_data.costs_a[0] + optimal_m1, 2),
            "strategy_2": round(market_data.costs_a[1] + optimal_m2, 2)
        },

        # Экономические показатели
        "expected_performance": {
            "demand": {
                "strategy_1": round(expected_demand_1, 2),
                "strategy_2": round(expected_demand_2, 2)
            },
            "profit": {
                "strategy_1": round(profit_1, 2),
                "strategy_2": round(profit_2, 2)
            }
        }
    }

    return result