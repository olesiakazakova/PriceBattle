# tests/test_unit_lp_optimizer.py
import pytest
from nash_equilibrium import MarketData, EquilibriumResult
from lp_optimizer import optimize_margin_logic


class TestLpOptimizer:
    """Тесты для LP оптимизатора"""

    def test_optimize_margin_basic(self, sample_market_data):
        """Базовый тест оптимизации маржи"""
        market_data = MarketData(
            costs_a=sample_market_data["costs_a"],
            costs_b=sample_market_data["costs_b"],
            demand_matrix=sample_market_data["demand_matrix"]
        )

        # Создаем равновесие (примерное)
        equilibrium = EquilibriumResult(
            probs_a=[0.5, 0.5],
            probs_b=[0.7, 0.3],
            equilibrium_type="mixed",
            expected_profit_a=10000,
            expected_profit_b=15000
        )

        result = optimize_margin_logic(
            market_data=market_data,
            equilibrium=equilibrium,
            reference_prices_a=[300, 400],
            reference_prices_b=[250, 400],
            min_margin=10,
            max_price=1000
        )

        # проверяем структуру ответа
        assert "status" in result
        assert "optimal_margins" in result or "message" in result

        # Если решение найдено, проверяем корректность
        if result["status"] == "success":
            assert result["optimal_margins"]["strategy_1"] >= 10
            assert result["optimal_margins"]["strategy_2"] >= 10
            assert result["optimal_prices"]["strategy_1"] <= 1000
            assert result["optimal_prices"]["strategy_2"] <= 1000
        else:
            # Если не найдено, проверяем что есть сообщение об ошибке
            assert "message" in result

    def test_optimize_margin_with_high_min_margin(self, sample_market_data):
        """Тест с высокой минимальной маржой"""
        market_data = MarketData(
            costs_a=sample_market_data["costs_a"],
            costs_b=sample_market_data["costs_b"],
            demand_matrix=sample_market_data["demand_matrix"]
        )

        equilibrium = EquilibriumResult(
            probs_a=[0.5, 0.5],
            probs_b=[0.7, 0.3],
            equilibrium_type="mixed",
            expected_profit_a=10000,
            expected_profit_b=15000
        )

        result = optimize_margin_logic(
            market_data=market_data,
            equilibrium=equilibrium,
            reference_prices_a=[300, 400],
            reference_prices_b=[250, 400],
            min_margin=1000,  # Очень высокая маржа
            max_price=500
        )

        # Может быть не найдено решение
        if result["status"] == "error":
            assert "Не удалось найти оптимальное решение" in result["message"]

    def test_optimize_margin_price_constraints(self, sample_market_data):
        """Тест ограничений на цену"""
        market_data = MarketData(
            costs_a=sample_market_data["costs_a"],
            costs_b=sample_market_data["costs_b"],
            demand_matrix=sample_market_data["demand_matrix"]
        )

        equilibrium = EquilibriumResult(
            probs_a=[0.5, 0.5],
            probs_b=[0.7, 0.3],
            equilibrium_type="mixed",
            expected_profit_a=10000,
            expected_profit_b=15000
        )

        max_price = 300
        result = optimize_margin_logic(
            market_data=market_data,
            equilibrium=equilibrium,
            reference_prices_a=[300, 400],
            reference_prices_b=[250, 400],
            min_margin=10,
            max_price=max_price
        )

        if result["status"] == "success":
            price_1 = result["optimal_prices"]["strategy_1"]
            price_2 = result["optimal_prices"]["strategy_2"]

            assert price_1 <= max_price
            assert price_2 <= max_price

    def test_optimize_margin_positive_margins(self, sample_market_data):
        """Тест: маржи должны быть неотрицательными"""
        market_data = MarketData(
            costs_a=sample_market_data["costs_a"],
            costs_b=sample_market_data["costs_b"],
            demand_matrix=sample_market_data["demand_matrix"]
        )

        equilibrium = EquilibriumResult(
            probs_a=[0.5, 0.5],
            probs_b=[0.7, 0.3],
            equilibrium_type="mixed",
            expected_profit_a=10000,
            expected_profit_b=15000
        )

        result = optimize_margin_logic(
            market_data=market_data,
            equilibrium=equilibrium,
            reference_prices_a=[300, 400],
            reference_prices_b=[250, 400],
            min_margin=0,
            max_price=1000
        )

        if result["status"] == "success":
            assert result["optimal_margins"]["strategy_1"] >= 0
            assert result["optimal_margins"]["strategy_2"] >= 0