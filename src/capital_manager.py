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

    def reconcile(self, allocations: dict[str, float]):
        cleaned = {
            str(strategy): max(0.0, float(amount))
            for strategy, amount in allocations.items()
            if float(amount) > 0
        }
        self._allocated = cleaned

    def kelly_size(
        self,
        strategy: str,
        win_prob: float,
        net_return_pct: float,
        kelly_fraction: float = 0.25,
        min_size: float = 1.0,
        max_size: float = 300.0,
    ) -> float:
        """
        Fractional Kelly criterion bet sizing.

        win_prob       — estimated probability of winning (0–1)
        net_return_pct — net profit per dollar risked if we win (e.g. 0.06 for 6%)
        kelly_fraction — safety multiplier; 0.25 = quarter-Kelly (recommended)

        Kelly formula: f* = (b·p − q) / b
          where b = net odds (net_return_pct), p = win_prob, q = 1 − p
        """
        if win_prob <= 0 or net_return_pct <= 0:
            return min_size

        b = net_return_pct
        p = min(win_prob, 0.9999)
        q = 1.0 - p

        kelly_f = (b * p - q) / b
        kelly_f = max(0.0, kelly_f) * kelly_fraction

        bet = kelly_f * self.total_usdc
        available = self.strategy_available(strategy)
        return round(max(min_size, min(bet, max_size, available)), 2)

    def summary(self) -> dict:
        return {
            "total_usdc": round(self.total_usdc, 2),
            "available_usdc": round(self.available_usdc, 2),
            "reserve_usdc": round(self.total_usdc * self.reserve_pct / 100, 2),
            "allocated": {k: round(v, 2) for k, v in self._allocated.items()},
        }
