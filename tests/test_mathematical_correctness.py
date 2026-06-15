# tests/test_mathematical_correctness.py
import pytest
import numpy as np
from nash_equilibrium import MarketData, GameState


class TestMathematicalCorrectness:
    """Тесты математической корректности"""

    def test_nash_equilibrium_pure_strategy(self):
        """Тест: чистое равновесие в игре "Дилемма заключенного" типа"""
        # Создаем игру с доминирующей стратегией
        market_data = MarketData(
            costs_a=[100, 100],
            costs_b=[100, 100],
            demand_matrix=[[200, 50], [180, 30]]
        )

        state = GameState()
        state.set_market_data(
            market_data,
            reference_prices_a=[200, 220],
            reference_prices_b=[200, 220]
        )

        eq = state.get_equilibrium()

        # В такой игре должно быть чистое равновесие
        assert eq.equilibrium_type in ["pure_1", "pure_2"]

    def test_nash_equilibrium_mixed_strategy(self, sample_market_data):
        """Тест: смешанное равновесие"""
        market_data = MarketData(
            costs_a=sample_market_data["costs_a"],
            costs_b=sample_market_data["costs_b"],
            demand_matrix=sample_market_data["demand_matrix"]
        )

        state = GameState()
        state.set_market_data(
            market_data,
            sample_market_data["reference_prices_a"],
            sample_market_data["reference_prices_b"]
        )

        eq = state.get_equilibrium()

        # Проверяем, что вероятности между 0 и 1
        for p in eq.probs_a:
            assert 0 <= p <= 1
        for q in eq.probs_b:
            assert 0 <= q <= 1

    def test_expected_profit_calculation(self, sample_market_data):
        """Тест: проверка расчета ожидаемой прибыли"""
        market_data = MarketData(
            costs_a=sample_market_data["costs_a"],
            costs_b=sample_market_data["costs_b"],
            demand_matrix=sample_market_data["demand_matrix"]
        )

        prices_a = [300, 400]
        prices_b = [250, 400]

        profit_a, profit_b = market_data.calculate_profit_matrices(prices_a, prices_b)

        # Ручной расчет прибыли для стратегии (0,0)
        # (Цена_A - Себестоимость_A) * Спрос = (300-200) * 150 = 15000
        assert profit_a[0][0] == 15000
        assert profit_b[0][0] == 22500

        # Расчет для стратегии (0,1)
        # A: (300-200) * 70 = 7000
        # B: (400-300) * 70 = 7000
        assert profit_a[0][1] == 7000
        assert profit_b[0][1] == 7000

    def test_profit_non_negative(self):
        """Тест: прибыль не может быть отрицательной при разумных ценах"""
        market_data = MarketData(
            costs_a=[200, 300],
            costs_b=[200, 300],
            demand_matrix=[[100, 80], [80, 100]]
        )

        # Цены выше себестоимости
        prices_a = [300, 400]
        prices_b = [300, 400]

        profit_a, profit_b = market_data.calculate_profit_matrices(prices_a, prices_b)

        for i in range(2):
            for j in range(2):
                assert profit_a[i][j] >= 0
                assert profit_b[i][j] >= 0

    def test_linear_programming_optimality(self):
        """Тест: проверка оптимальности решения ЛП"""
        from pulp import LpProblem, LpMaximize, LpVariable, value

        # Простая задача оптимизации
        prob = LpProblem("Test", LpMaximize)
        x = LpVariable("x", lowBound=0)
        y = LpVariable("y", lowBound=0)

        prob += x + y  # Максимизируем сумму
        prob += x <= 10
        prob += y <= 10
        prob += x + y <= 15

        prob.solve()

        # Получаем значения
        x_val = value(x)
        y_val = value(y)
        optimal_value = value(prob.objective)

        # Проверяем, что целевая функция достигла максимума (15)
        assert abs(optimal_value - 15) < 1e-6, f"Оптимальное значение должно быть 15, получено {optimal_value}"

        # Проверяем, что решение удовлетворяет ВСЕМ ограничениям
        assert x_val >= 0, f"x = {x_val} должно быть >= 0"
        assert y_val >= 0, f"y = {y_val} должно быть >= 0"
        assert x_val <= 10 + 1e-6, f"x = {x_val} должно быть <= 10"
        assert y_val <= 10 + 1e-6, f"y = {y_val} должно быть <= 10"
        assert x_val + y_val <= 15 + 1e-6, f"x+y = {x_val + y_val} должно быть <= 15"

        # Проверяем, что решение действительно достигает границы (сумма = 15)
        assert abs(x_val + y_val - 15) < 1e-6, f"Сумма x+y должна быть 15, получено {x_val + y_val}"

        # Проверяем, что решение находится на границе допустимой области
        # Любая комбинация (x, y) с x+y=15, где 5 ≤ x ≤ 10, 5 ≤ y ≤ 10 является оптимальной
        assert 5 <= x_val <= 10, f"x = {x_val} должен быть между 5 и 10"
        assert 5 <= y_val <= 10, f"y = {y_val} должен быть между 5 и 10"

    def test_symmetric_game_equilibrium(self, symmetric_market_data):
        """Тест: в симметричной игре равновесие должно быть симметричным"""
        market_data = MarketData(
            costs_a=symmetric_market_data["costs_a"],
            costs_b=symmetric_market_data["costs_b"],
            demand_matrix=symmetric_market_data["demand_matrix"]
        )

        state = GameState()
        state.set_market_data(
            market_data,
            symmetric_market_data["reference_prices_a"],
            symmetric_market_data["reference_prices_b"]
        )

        eq = state.get_equilibrium()

        # В симметричной игре вероятности могут быть близки
        assert abs(eq.probs_a[0] - eq.probs_b[0]) < 0.1