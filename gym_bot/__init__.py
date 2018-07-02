import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Bot-v0',
    entry_point='gym_bot.envs:BotEnv',
)
