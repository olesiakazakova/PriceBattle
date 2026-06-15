from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional
import nashpy as nash
import numpy as np
import pulp


# ========== МОДЕЛИ ДАННЫХ ==========
class PayoffMatrix(BaseModel):
    payoff_a: List[List[float]]
    payoff_b: List[List[float]]
    prices_a: List[float] = Field(..., min_length=2)
    prices_b: List[float] = Field(..., min_length=2)

    @model_validator(mode='after')
    def validate(self):
        n, m = len(self.prices_a), len(self.prices_b)
        if len(self.payoff_a) != n or len(self.payoff_b) != n:
            raise ValueError('Неверное количество строк')
        for row in self.payoff_a + self.payoff_b:
            if len(row) != m:
                raise ValueError('Неверное количество столбцов')
        return self


class EqResponse(BaseModel):
    strat_a: List[float]
    strat_b: List[float]
    payoff_a: float
    payoff_b: float


class SimulationRequest(BaseModel):
    matrix: PayoffMatrix
    strategy_a: List[float]  # вероятности выбора цен для игрока A
    strategy_b: List[float]  # вероятности выбора цен для игрока B
    iterations: int = Field(1000, ge=1, le=10000)  # количество симуляций


class SimulationResponse(BaseModel):
    expected_payoff_a: float
    expected_payoff_b: float
    std_payoff_a: float
    std_payoff_b: float
    most_common_outcome: dict
    win_rate_a: float  # вероятность что A выиграл (payoff_a > payoff_b)
    win_rate_b: float  # вероятность что B выиграл
    draw_rate: float  # ничья


class MinMarginResponse(BaseModel):
    strategy: List[float]
    margin: float


# ========== СЕРВИСЫ ==========
def find_equilibrium(matrix: PayoffMatrix):
    A, B = np.array(matrix.payoff_a), np.array(matrix.payoff_b)
    game = nash.Game(A, B)
    eqs = list(game.support_enumeration())
    if not eqs:
        return None
    sa, sb = eqs[0]
    return EqResponse(
        strat_a=[round(float(p), 6) for p in sa],
        strat_b=[round(float(p), 6) for p in sb],
        payoff_a=float(sa @ A @ sb),
        payoff_b=float(sa @ B @ sb)
    )


def min_margin(matrix: PayoffMatrix, player: str):
    n, m = len(matrix.prices_a), len(matrix.prices_b)

    if player == 'A':
        payoff = np.array(matrix.payoff_a)
        prices = matrix.prices_a
        n_strat = n
        opp_strat = m
    else:
        # Для игрока B транспонируем, чтобы строки были стратегиями B
        payoff = np.array(matrix.payoff_b).T
        prices = matrix.prices_b
        n_strat = m
        opp_strat = n

    prob = pulp.LpProblem("MinMargin", pulp.LpMaximize)
    margin = pulp.LpVariable("margin", lowBound=0)
    x = [pulp.LpVariable(f"x_{i}", 0, 1) for i in range(n_strat)]

    prob += margin
    prob += pulp.lpSum(x) == 1

    if player == 'A':
        for j in range(opp_strat):
            prob += pulp.lpSum(x[i] * payoff[i, j] for i in range(n_strat)) >= margin * pulp.lpSum(
                x[i] * prices[i] for i in range(n_strat))
    else:
        for i in range(opp_strat):
            prob += pulp.lpSum(x[j] * payoff[j, i] for j in range(n_strat)) >= margin * pulp.lpSum(
                x[j] * prices[j] for j in range(n_strat))

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if prob.status == pulp.LpOptimal:
        return [pulp.value(x[i]) for i in range(n_strat)], pulp.value(margin)
    return None, None


def simulate_game(matrix: PayoffMatrix, strategy_a: List[float], strategy_b: List[float], iterations: int):
    """Симуляция игры с заданными смешанными стратегиями"""

    # Проверка валидности стратегий
    if abs(sum(strategy_a) - 1.0) > 1e-6 or abs(sum(strategy_b) - 1.0) > 1e-6:
        raise ValueError("Стратегии должны суммироваться в 1")

    if len(strategy_a) != len(matrix.prices_a) or len(strategy_b) != len(matrix.prices_b):
        raise ValueError("Размерность стратегий не соответствует ценам")

    payoffs_a = []
    payoffs_b = []
    outcomes = {}

    for _ in range(iterations):
        # Выбор действий на основе вероятностей
        action_a = np.random.choice(len(strategy_a), p=strategy_a)
        action_b = np.random.choice(len(strategy_b), p=strategy_b)

        payoff_a = matrix.payoff_a[action_a][action_b]
        payoff_b = matrix.payoff_b[action_a][action_b]

        payoffs_a.append(payoff_a)
        payoffs_b.append(payoff_b)

        # Фиксация исхода
        outcome_key = (matrix.prices_a[action_a], matrix.prices_b[action_b])
        outcomes[outcome_key] = outcomes.get(outcome_key, 0) + 1

    # Вычисление статистики
    payoffs_a = np.array(payoffs_a)
    payoffs_b = np.array(payoffs_b)

    # Кто выиграл
    wins_a = np.sum(payoffs_a > payoffs_b)
    wins_b = np.sum(payoffs_b > payoffs_a)
    draws = np.sum(payoffs_a == payoffs_b)

    # Самый частый исход
    most_common = max(outcomes.items(), key=lambda x: x[1])

    return SimulationResponse(
        expected_payoff_a=float(np.mean(payoffs_a)),
        expected_payoff_b=float(np.mean(payoffs_b)),
        std_payoff_a=float(np.std(payoffs_a)),
        std_payoff_b=float(np.std(payoffs_b)),
        most_common_outcome={
            "price_a": most_common[0][0],
            "price_b": most_common[0][1],
            "frequency": most_common[1] / iterations
        },
        win_rate_a=wins_a / iterations,
        win_rate_b=wins_b / iterations,
        draw_rate=draws / iterations
    )


# ========== API ==========
app = FastAPI(
    title="PriceBattle Mini",
    description="Симулятор конкурентного ценообразования с поиском равновесия и оптимизацией маржи",
    version="2.0"
)


@app.post("/equilibrium", response_model=EqResponse)
async def equilibrium(matrix: PayoffMatrix):
    """Найти равновесие Нэша в смешанных стратегиях"""
    result = find_equilibrium(matrix)
    if not result:
        raise HTTPException(404, "Равновесие не найдено")
    return result


@app.post("/min_margin", response_model=MinMarginResponse)
async def min_margin_api(matrix: PayoffMatrix, player: str = "A"):
    """Найти стратегию, максимизирующую минимальную маржу для игрока"""
    if player not in ["A", "B"]:
        raise HTTPException(400, "player должен быть 'A' или 'B'")

    strat, margin = min_margin(matrix, player)
    if strat is None:
        raise HTTPException(400, "Нет решения для задачи минимизации маржи")
    return {"strategy": strat, "margin": margin}


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(req: SimulationRequest):
    """Симуляция игры с заданными смешанными стратегиями"""
    try:
        result = simulate_game(
            req.matrix,
            req.strategy_a,
            req.strategy_b,
            req.iterations
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/simulate_equilibrium")
async def simulate_equilibrium(matrix: PayoffMatrix, iterations: int = 1000):
    """Найти равновесие и сразу его просимулировать"""
    eq = find_equilibrium(matrix)
    if not eq:
        raise HTTPException(404, "Равновесие не найдено")

    simulation = simulate_game(matrix, eq.strat_a, eq.strat_b, iterations)

    return {
        "equilibrium": eq,
        "simulation": simulation
    }


@app.get("/health")
async def health():
    """Проверка работоспособности сервиса"""
    return {"status": "ok",
            "endpoints": ["/equilibrium", "/min_margin", "/simulate", "/simulate_equilibrium", "/health"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)