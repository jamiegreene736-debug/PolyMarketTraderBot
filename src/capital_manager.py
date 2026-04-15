from loguru import logger


class CapitalManager:
    """
    Tracks capital allocation across strategies.
    Ensures no strategy exceeds its configured capital percentage
    and a reserve is always maintained.
    """

    def __init__(self, total_usdc: float, strategy_config: dict, reserve_pct: float = 10.0):
        self.total_usdc = total_usdc
        self.reserve_pct = reserve_pct
        self.strategy_config = strategy_config
        self._allocated: dict[str, float] = {}

    def update_balance(self, new_balance: float):
        self.total_usdc = new_balance

    @property
    def available_usdc(self) -> float:
        reserve = self.total_usdc * (self.reserve_pct / 100)
        tradeable = self.total_usdc - reserve
        total_allocated = sum(self._allocated.values())
        return max(0.0, tradeable - total_allocated)

    def strategy_limit(self, strategy: str) -> float:
        pct = self.strategy_config.get(strategy, {}).get("capital_pct", 0)
        return self.total_usdc * (pct / 100)

    def strategy_available(self, strategy: str) -> float:
        limit = self.strategy_limit(strategy)
        used = self._allocated.get(strategy, 0.0)
        return max(0.0, min(limit - used, self.available_usdc))

    def can_allocate(self, strategy: str, amount: float) -> bool:
        return self.strategy_available(strategy) >= amount

    def allocate(self, strategy: str, amount: float) -> bool:
        if not self.can_allocate(strategy, amount):
            logger.warning(f"Capital limit reached for {strategy}: "
                           f"available=${self.strategy_available(strategy):.2f}, requested=${amount:.2f}")
            return False
        self._allocated[strategy] = self._allocated.get(strategy, 0.0) + amount
        logger.debug(f"Allocated ${amount:.2f} to {strategy} "
                     f"(total={self._allocated[strategy]:.2f}/{self.strategy_limit(strategy):.2f})")
        return True

    def release(self, strategy: str, amount: float):
        current = self._allocated.get(strategy, 0.0)
        self._allocated[strategy] = max(0.0, current - amount)

    def summary(self) -> dict:
        return {
            "total_usdc": round(self.total_usdc, 2),
            "available_usdc": round(self.available_usdc, 2),
            "reserve_usdc": round(self.total_usdc * self.reserve_pct / 100, 2),
            "allocated": {k: round(v, 2) for k, v in self._allocated.items()},
        }
