from src.strategies.base import BaseStrategy


class CrossPlatformArbStrategy(BaseStrategy):
    """
    Cross-Platform Arbitrage Strategy (DISABLED)
    ---------------------------------------------
    Detects price differences for the same event between Polymarket.us
    and other platforms (e.g. Kalshi).

    To enable:
    1. Set enabled: true in config.yaml
    2. Add KALSHI_API_KEY and KALSHI_SECRET to your .env
    3. Implement Kalshi client in src/kalshi_client.py
    """

    async def run(self):
        if not self.enabled:
            return
        self.log(
            "Cross-platform arb requires Kalshi API integration. "
            "See src/strategies/cross_platform.py for setup instructions.",
            level="warning"
        )
