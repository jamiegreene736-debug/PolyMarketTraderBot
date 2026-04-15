from abc import ABC, abstractmethod
from loguru import logger
from src.client import PolymarketClient
from src.market_data import MarketData
from src.order_manager import OrderManager
from src.capital_manager import CapitalManager


class BaseStrategy(ABC):
    def __init__(self, name: str, config: dict,
                 client: PolymarketClient,
                 market_data: MarketData,
                 order_manager: OrderManager,
                 capital_manager: CapitalManager):
        self.name = name
        self.config = config
        self.client = client
        self.market_data = market_data
        self.order_manager = order_manager
        self.capital_manager = capital_manager
        self.enabled = config.get("enabled", False)

    @abstractmethod
    async def run(self):
        pass

    def log(self, msg: str, level: str = "info"):
        getattr(logger, level)(f"[{self.name}] {msg}")
