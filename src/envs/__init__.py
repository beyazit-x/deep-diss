from gym.envs.registration import register

from envs.gym_letters.letter_env import LetterEnv
from envs.gym_letters.simple_ltl_env import SimpleLTLEnv
from envs.gym_letters.simple_dfa_env import SimpleDFAEnv
from envs.minigrid.minigrid_env import MinigridEnv
from envs.safety.zones_env import ZonesEnv
from envs.gridworld.gridworld_env import GridworldEnv

__all__ = ["LetterEnv", "SimpleLTLEnv", "SimpleDFAEnv", "MinigridEnv", "ZonesEnv"]
# __all__ = ["LetterEnv", "SimpleLTLEnv", "MinigridEnv"]


### Simple LTL Envs
register(
    id='Simple-LTL-Env-v0',
    entry_point='envs.gym_letters.simple_ltl_env:SimpleLTLEnvDefault'
)

register(
    id='Simple-DFA-Env-v0',
    entry_point='envs.gym_letters.simple_dfa_env:SimpleDFAEnvDefault'
)

### Letter Envs
register(
    id='Letter-4x4-v0',
    entry_point='envs.gym_letters.letter_env:LetterEnv4x4'
)

register(
    id='Letter-4x4-v1',
    entry_point='envs.gym_letters.letter_env:LetterEnvFixedMap4x4'
)

register(
    id='Letter-5x5-v0',
    entry_point='envs.gym_letters.letter_env:LetterEnv5x5'
)

register(
    id='Letter-5x5-v1',
    entry_point='envs.gym_letters.letter_env:LetterEnvFixedMap5x5'
)

register(
    id='Letter-5x5-v2',
    entry_point='envs.gym_letters.letter_env:LetterEnvAgentCentric5x5'
)

register(
    id='Letter-5x5-v3',
    entry_point='envs.gym_letters.letter_env:LetterEnvAgentCentricFixedMap5x5'
)

register(
    id='Letter-5x5-v4',
    entry_point='envs.gym_letters.letter_env:LetterEnvShortAgentCentric5x5'
)

register(
    id='Letter-5x5-v5',
    entry_point='envs.gym_letters.letter_env:LetterEnvShortAgentCentricFixedMap5x5'
)

register(
    id='Letter-7x7-v0',
    entry_point='envs.gym_letters.letter_env:LetterEnv7x7'
)

register(
    id='Letter-7x7-v1',
    entry_point='envs.gym_letters.letter_env:LetterEnvFixedMap7x7'
)

register(
    id='Letter-7x7-v2',
    entry_point='envs.gym_letters.letter_env:LetterEnvAgentCentric7x7'
)

register(
    id='Letter-7x7-v3',
    entry_point='envs.gym_letters.letter_env:LetterEnvAgentCentricFixedMap7x7'
)

register(
    id='Letter-7x7-v4',
    entry_point='envs.gym_letters.letter_env:LetterEnvAgentCentricFixedMapShort7x7'
)

register(
    id='Letter-7x7-v5',
    entry_point='envs.gym_letters.letter_env:LetterEnvAgentCentricFixedMapMid7x7'
)

### Minigrid Envs
register(
    id='Adversarial-v0',
    entry_point='envs.minigrid.minigrid_env:AdversarialMinigridEnv'
)

### Safety Envs
register(
    id='Zones-1-v0',
    entry_point='envs.safety.zones_env:ZonesEnv1')

register(
    id='Zones-1-v1',
    entry_point='envs.safety.zones_env:ZonesEnv1Fixed')

register(
    id='Zones-5-v0',
    entry_point='envs.safety.zones_env:ZonesEnv5')

register(
    id='Zones-5-v1',
    entry_point='envs.safety.zones_env:ZonesEnv5Fixed')

### Gridworld
register(
    id='Gridworld-v1',
    entry_point='envs.gridworld.gridworld_env:GridworldEnv1')

register(
    id='Gridworld-v2',
    entry_point='envs.gridworld.gridworld_env:GridworldEnv2')

register(
    id='Gridworld-v3',
    entry_point='envs.gridworld.gridworld_env:GridworldEnv3')

register(
    id='DFADummy-v0',
    entry_point='envs.dfa_world.dfa_world:DFADummyEnv')

