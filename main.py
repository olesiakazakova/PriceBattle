"""
PriceBattle - симулятор конкурентного ценообразования
Темы: Матричные игры + ЛП
Краткое описание:
Два продавца выбирают цены.
Платежи задаются таблицей спроса/прибыли.
Поиск равновесия в смешанных стратегиях + оптимизация минимальной маржи через ЛП.
Технологии: FastAPI, nashpy, PuLP, Pydantic
"""

import logging
import traceback
from typing import List, Optional

from fastapi import FastAPI, HTTPException
import uvicorn

from pydantic import BaseModel, Field, validator

from nash_equilibrium import GameState, EquilibriumResult
from lp_optimizer import MarginOptimizationRequest, optimize_margin_logic

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== Pydantic схемы данных ==========

class MarketDataRequest(BaseModel):
    """Запрос на загрузку рыночных данных"""

    costs_a: List[float] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="Себестоимость товара для игрока A (продавец 1). "
                    "Первый элемент — для стратегии 1, второй — для стратегии 2.",
        example=[200.0, 350.0]
    )

    costs_b: List[float] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="Себестоимость товара для игрока B (продавец 2). "
                    "Первый элемент — для стратегии 1, второй — для стратегии 2.",
        example=[100.0, 300.0]
    )

    demand_matrix: List[List[float]] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="Матрица спроса 2×2. "
                    "demand_matrix[i][j] — спрос, когда A выбирает стратегию i, а B — стратегию j. "
                    "Формат: [[d11, d12], [d21, d22]]",
        example=[[150, 70], [150, 70]]
    )

    reference_prices_a: List[float] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="Цены игрока A, при которых измерен спрос. "
                    "Первый элемент — цена для стратегии 1, второй — для стратегии 2.",
        example=[300.0, 400.0]
    )

    reference_prices_b: List[float] = Field(
        ...,
        min_items=2,
        max_items=2,
        description="Цены игрока B, при которых измерен спрос. "
                    "Первый элемент — цена для стратегии 1, второй — для стратегии 2.",
        example=[250.0, 400.0]
    )

    # Дополнительная валидация
    @validator('costs_a', 'costs_b')
    def validate_costs(cls, v):
        if any(cost < 0 for cost in v):
            raise ValueError('Себестоимость не может быть отрицательной')
        return v

    @validator('demand_matrix')
    def validate_demand_matrix(cls, v):
        if len(v) != 2:
            raise ValueError('demand_matrix должна иметь 2 строки')
        if len(v[0]) != 2 or len(v[1]) != 2:
            raise ValueError('Каждая строка demand_matrix должна иметь 2 элемента')
        if any(d < 0 for row in v for d in row):
            raise ValueError('Спрос не может быть отрицательным')
        return v

    @validator('reference_prices_a', 'reference_prices_b')
    def validate_prices(cls, v):
        if any(price < 0 for price in v):
            raise ValueError('Цены не могут быть отрицательными')
        return v


# ========== Инициализация приложения ==========

game_state = GameState()
app = FastAPI(title="PriceBattle API", description="Симулятор конкурентного ценообразования")


# ========== Эндпоинт 1: Загрузка рыночных данных ==========

@app.post("/game/set_market_data")
async def set_market_data(request: MarketDataRequest):
    """
    ## Загрузить рыночные данные

    **Этот эндпоинт должен быть вызван ПЕРВЫМ** перед любыми другими операциями.

    ## Что происходит:
    1. Валидация всех входных данных
    2. Сохранение в глобальное состояние GameState
    3. Расчёт матриц прибыли при референсных ценах
    4. Сброс кэша равновесия

    **Пример использования:**
    ```json
    {
        "costs_a": [200, 350],
        "costs_b": [100, 300],
        "demand_matrix": [[150, 70], [150, 70]],
        "reference_prices_a": [300, 400],
        "reference_prices_b": [250, 400]
    }
    """
    try:
        from nash_equilibrium import MarketData

        # Создаём внутреннюю модель MarketData из запроса
        market_data = MarketData(
            costs_a=request.costs_a,
            costs_b=request.costs_b,
            demand_matrix=request.demand_matrix
        )

        # Сохраняем в глобальное состояние
        game_state.set_market_data(
            market_data,
            request.reference_prices_a,
            request.reference_prices_b
        )

        # Рассчитываем матрицы прибыли при референсных ценах
        profit_a, profit_b = market_data.calculate_profit_matrices(
            request.reference_prices_a,
            request.reference_prices_b
        )

        return {
            "status": "ok",
            "message": "Рыночные данные успешно загружены",
            "profit_matrices_at_reference_prices": {
                "player_a": profit_a,
                "player_b": profit_b
            },
        }

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(400, str(e))


# ========== Эндпоинт 2: Поиск равновесия Нэша ==========

@app.post("/game/equilibrium", response_model=EquilibriumResult)
async def find_equilibrium():
    """
    ## Найти равновесие Нэша на основе загруженных рыночных данных

    ## Описание

    Эндпоинт возвращает равновесие Нэша (чистое или смешанное) для двух продавцов
    на основе ранее загруженных рыночных данных.

    **Важно:** перед вызовом этого эндпоинта необходимо загрузить рыночные данные
    через `POST /game/set_market_data`.

    ## Типы равновесия

    - **mixed** - смешанное равновесие: оба игрока используют обе стратегии с ненулевыми вероятностями
    - **pure_1** - чистое равновесие: оба игрока выбирают стратегию 1
    - **pure_2** - чистое равновесие: оба игрока выбирают стратегию 2
    """
    try:
        return game_state.get_equilibrium()
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        raise HTTPException(500, str(e))


# ========== Эндпоинт 3: Оптимизация маржи ==========

@app.post("/game/margin_optimize")
async def optimize_margin(request: MarginOptimizationRequest):
    """
    ## Оптимизация минимальной маржи через линейное программирование для игрока А

    ## Описание

    Эндпоинт решает задачу **max-min оптимизации**: максимизирует минимальную маржу
    среди двух стратегий игрока A.

    **Пример использования:**
    ```json
    {
        "min_margin": 100,
        "max_price": 500
    }
    """
    try:
        logger.info("=" * 50)
        logger.info("НАЧАЛО ОПТИМИЗАЦИИ МАРЖИ")
        logger.info("=" * 50)

        # Получаем рыночные данные и равновесие
        market_data = game_state.get_market_data()
        equilibrium = game_state.get_equilibrium()
        reference_prices_a = game_state.get_reference_prices_a()
        reference_prices_b = game_state.get_reference_prices_b()

        # Вызываем логику оптимизации
        result = optimize_margin_logic(
            market_data=market_data,
            equilibrium=equilibrium,
            reference_prices_a=reference_prices_a,
            reference_prices_b=reference_prices_b,
            min_margin=request.min_margin,
            max_price=request.max_price
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Внутренняя ошибка: {str(e)}"}


# ========== Эндпоинт 4: Сброс состояния ==========

@app.post("/game/clear")
async def clear_state():
    """## Сбросить загруженные данные"""
    game_state.market_data = None
    game_state._equilibrium_cache = None
    return {"status": "cleared"}


# ========== Эндпоинт 5: Статус ==========

@app.get("/game/status")
async def get_status():
    """## Получить текущий статус"""
    if game_state.market_data is None:
        return {"status": "no_data"}

    return {
        "status": "ready",
        "costs_a": game_state.market_data.costs_a,
        "costs_b": game_state.market_data.costs_b,
        "demand_matrix": game_state.market_data.demand_matrix
    }


# ========== Запуск сервера ==========

if __name__ == "__main__":
    print("=" * 60)
    print("PriceBattle API v1.0")
    print("=" * 60)
    print("\nДоступные эндпоинты:")
    print("  POST /game/set_market_data - загрузка рыночных данных")
    print("  POST /game/equilibrium - поиск равновесия")
    print("  POST /game/margin_optimize - оптимизация маржи")
    print("  GET /game/status - статус")
    print("  POST /game/clear - сброс")
    print("\n" + "=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)