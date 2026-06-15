# tests/test_unit_nash_equilibrium.py
import pytest
import numpy as np
from nash_equilibrium import MarketData, GameState, EquilibriumResult


class TestMarketData:
    """Тесты для MarketData класса"""

    def test_market_data_creation(self, sample_market_data):
        """Тест создания объекта MarketData"""
        market_data = MarketData(
            costs_a=sample_market_data["costs_a"],
            costs_b=sample_market_data["costs_b"],
            demand_matrix=sample_market_data["demand_matrix"]
        )

        assert market_data.costs_a == [200.0, 350.0]
        assert market_data.costs_b == [100.0, 300.0]
        assert market_data.demand_matrix == [[150.0, 70.0], [150.0, 70.0]]

    def test_calculate_profit_matrices(self, sample_market_data):
        """Тест расчета матриц прибыли"""
        market_data = MarketData(
            costs_a=sample_market_data["costs_a"],
            costs_b=sample_market_data["costs_b"],
            demand_matrix=sample_market_data["demand_matrix"]
        )

        prices_a = [300.0, 400.0]
        prices_b = [250.0, 400.0]

        profit_a, profit_b = market_data.calculate_profit_matrices(prices_a, prices_b)

        # Проверяем прибыль A для пары стратегий (0,0)
        # (300 - 200) * 150 = 100 * 150 = 15000
        assert profit_a[0][0] == 15000.0

        # Проверяем прибыль B для пары стратегий (0,0)
        # (250 - 100) * 150 = 150 * 150 = 22500
        assert profit_b[0][0] == 22500.0

    def test_profit_matrices_shape(self, sample_market_data):
        """Тест размерности матриц прибыли"""
        market_data = MarketData(
            costs_a=sample_market_data["costs_a"],
            costs_b=sample_market_data["costs_b"],
            demand_matrix=sample_market_data["demand_matrix"]
        )

        profit_a, profit_b = market_data.calculate_profit_matrices(
            [300.0, 400.0], [250.0, 400.0]
        )

        assert len(profit_a) == 2
        assert len(profit_a[0]) == 2
        assert len(profit_b) == 2
        assert len(profit_b[0]) == 2

    def test_negative_costs_validation(self):
        """Тест валидации отрицательных себестоимостей"""
        with pytest.raises(ValueError, match="Себестоимость не может быть отрицательной"):
            MarketData(
                costs_a=[-100, 200],
                costs_b=[100, 200],
                demand_matrix=[[100, 80], [80, 100]]
            )


class TestGameState:
    """Тесты для GameState класса"""

    def test_set_market_data(self, sample_market_data, cleanup_state):
        """Тест загрузки рыночных данных"""
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

        assert state.market_data is not None
        assert state._reference_prices_a == [300.0, 400.0]
        assert state._reference_prices_b == [250.0, 400.0]

    def test_get_equilibrium_caching(self, sample_market_data, cleanup_state):
        """Тест кэширования равновесия"""
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

        eq1 = state.get_equilibrium()
        eq2 = state.get_equilibrium()

        # Должен вернуться тот же объект (кэширование)
        assert eq1 is eq2

    def test_equilibrium_probabilities_sum_to_one(self, sample_market_data, cleanup_state):
        """Тест: вероятности в равновесии суммируются в 1"""
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

        assert abs(sum(eq.probs_a) - 1.0) < 1e-6
        assert abs(sum(eq.probs_b) - 1.0) < 1e-6

    def test_equilibrium_profits_non_negative(self, sample_market_data, cleanup_state):
        """Тест: ожидаемая прибыль неотрицательна"""
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

        assert eq.expected_profit_a >= 0
        assert eq.expected_profit_b >= 0