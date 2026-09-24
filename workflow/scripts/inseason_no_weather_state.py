"""Matched-capacity removal of weather from the state-transition pathway."""
import torch

from forecast_bridge_state import ForecastState
from ndvi_tail_replacement import prefix_rollout


def without_weather(batch):
    return dict(batch, weather=torch.zeros_like(batch['weather']))


class NoWeatherState(ForecastState):
    def __init__(self, dim=128):
        super().__init__('biid', dim)

    def forward(self, batch, feedback=True):
        return super().forward(without_weather(batch), feedback=feedback)


@torch.no_grad()
def no_weather_prefix(model, batch, values, known):
    return prefix_rollout(model, without_weather(batch), values, known)
