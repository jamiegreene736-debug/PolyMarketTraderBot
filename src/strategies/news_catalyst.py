from src.strategies.base import BaseStrategy


class NewsCatalystStrategy(BaseStrategy):
    """
    News Catalyst Strategy (DISABLED)
    -----------------------------------
    Monitors real-time news feeds and places orders when a news event
    clearly resolves a prediction market before prices reprice.

    To enable:
    1. Set enabled: true in config.yaml
    2. Add NEWS_API_KEY to your .env (e.g. newsapi.org)
    3. Implement the news polling logic below

    High-value sources to integrate:
    - Sports APIs (ESPN, SportsRadar) for game results
    - Economic calendar APIs for Fed decisions, CPI data
    - AP/Reuters news webhooks for political events
    """

    async def run(self):
        if not self.enabled:
            return
        self.log(
            "News catalyst strategy requires a news API integration. "
            "See src/strategies/news_catalyst.py for setup instructions.",
            level="warning"
        )
