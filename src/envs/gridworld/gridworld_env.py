from __future__ import annotations

import gym
import numpy as np

from itertools import product
from typing import Any, Literal, Optional, Union

import attr
import funcy as fn

Player = Literal['ego', 'env']

Action = Literal['↑', '↓', '←', '→']
SLIP_DIRECTION = '←'
ACTION2VEC = {
    '→': (1, 0),
    '←': (-1, 0),
    '↑': (0, -1),
    '↓': (0, 1),
}


__all__ = [
    'Gridworld',
    'Action',
    'GridWorldNaive',
    'GridWorldState',
]

class GridworldEnv(gym.Env):
    """
    TODO : fill this out
    """

    def __init__(self, dim, start, overlay, slip_prob, end_ep_prob):
        """
            argument:
                -description
        """

        self.gw = GridWorldNaive(dim, start, overlay, slip_prob)
        self.dim = dim
        self.state = start
        self.end_ep_prob = end_ep_prob
        self.props = list(set(overlay.values()))
        self.props.remove('white')
        self.props.sort()

        # self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(dim), gym.spaces.Discrete(dim)))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(dim,dim), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.actions = list(ACTION2VEC.keys())

    def _get_obs(self):
        obs = np.zeros(shape=(self.dim,self.dim),dtype=np.uint8)
        obs[self.state.x-1,self.state.y-1] = 1
        return obs

    def step(self, action):
        """
        This function executes an action in the environment
        """

        action = self.actions[action]
        self.state = attr.evolve(self.state, action=action)
        moves = self.gw.moves(self.state)
        self.state = np.random.choice(list(moves.keys()), p=list(moves.values()))

        obs = self._get_obs()
        reward = 0.0
        done = np.random.random() < self.end_ep_prob

        return obs, reward, done, {}

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        """
        This function resets the world and collects the first observation.
        """

        self.state = self.gw.start
        obs = self._get_obs()

        return obs

    def get_events(self):
        loc = (self.state.x, self.state.y)
        color = self.gw.overlay.get(loc, '')
        if color == 'white': return []
        return [color]

    def get_propositions(self):
        return self.props


@attr.frozen
class GridWorldState:
    x: int
    y: int
    action: Optional[Action] = None

    def __repr__(self) -> str:
        if self.action is not None:
            return self.action
        return f'({self.x}, {self.y})'

    @property
    def succeed(self) -> GridWorldState:
        assert self.action is not None
        dx, dy = ACTION2VEC[self.action]
        return attr.evolve(self, x=self.x + dx, y=self.y + dy, action=None)

    @property
    def slip(self) -> GridWorldState:
        return attr.evolve(self, action=SLIP_DIRECTION).succeed

 
@attr.frozen
class GridWorldNaive:
    dim: int
    start: GridWorldState
    overlay: dict[tuple[int, int], str] = attr.ib(factory=dict)
    slip_prob: float = 1/32

    def sensor(self, state: Union[GridWorldState, tuple[int, int]]) -> Any:
        if isinstance(state, GridWorldState):
            if self.player(state) == 'env':
                return 'white'  # Ignore environment nodes.
            state = (state.x, state.y)
        return self.overlay.get(state, 'white')

    def clip(self, state: GridWorldState) -> GridWorldState:
        x = min(self.dim, max(1, state.x))
        y = min(self.dim, max(1, state.y))
        return attr.evolve(state, x=x, y=y)

    def moves(self, state: GridWorldState) -> dict[GridWorldState, float]:
        if self.player(state) == 'ego':
            moves = (attr.evolve(state, action=a) for a in ACTION2VEC)
            moves = (m for m in moves if self.clip(m.succeed) == m.succeed)
            return frozenset(moves)

        succeed, slip = self.clip(state.succeed), self.clip(state.slip)
        if state.action == SLIP_DIRECTION or succeed == slip:
            return {succeed: 1}
        return {succeed: 1 - self.slip_prob, slip: self.slip_prob}

    def player(self, state: GridWorldState) -> Player:
        return 'ego' if state.action is None else 'env'

    def to_string(self, state: GridWorldState) -> str:
        from blessings import Terminal  # type: ignore
        term = Terminal()
        buff = ''
        ego = 'x' if state.action is None else state.action

        def tile(point: tuple[int, int]) -> str:
            content = f' {ego} ' if point == (state.x, state.y) else '   '
            color = self.sensor(point)
            return getattr(term, f'on_{color}')(content)  # type: ignore

        for y in range(1, 1 + self.dim):
            row = ((x, y) for x in range(1, 1 + self.dim))
            buff += ''.join(map(tile, row)) + '\n'
        return buff

    @staticmethod
    def from_string(buff, start, codec, slip_prob=1/32) -> GridWorldNaive:
        overlay = {}
        rows = buff.split()
        widths = {len(row) for row in rows}
        assert len(widths) == 1
        dim = fn.first(widths)
        assert len(rows) == dim
        
        for y, row in enumerate(rows):
            aps = (codec.get(s, 'white') for s in row)
            overlay.update({(x+ 1, y + 1): ap for x, ap in enumerate(aps)}) 

        return GridWorldNaive(
            dim=dim, start=start,
            overlay=overlay, slip_prob=slip_prob
        )

    def path(self, seq):
        path = []
        for curr, prev in fn.with_prev(seq):
            if isinstance(curr, str):
                state, player = GridWorldState(*prev, action=curr), 'env'
            else:
                state, player = GridWorldState(*curr), 'ego'
            path.append((state, player))
        return path

class GridworldEnv1(GridworldEnv):
    def __init__(self):
        super().__init__(dim=3,
                         start=GridWorldState(x=3, y=1),
                         overlay={
                           (1, 1): 'yellow',
                           (1, 2): 'green',
                           (1, 3): 'green',
                           (2, 3): 'red',
                           (3, 2): 'blue',
                           (3, 3): 'blue',
                         },
                         slip_prob = 1/32,
                         end_ep_prob = 1/10
        )

class GridworldEnv2(GridworldEnv):
    def __init__(self):
        codec = {'y':'yellow', 'g':'green','r':'red','b':'blue'}
        buff="""y....g..
            ........
            .b.b...r
            .b.b...r
            .b.b....
            .b.b....
            rrrrrr.r
            g.y....."""
        start=GridWorldState(x=3, y=5)
        slip_prob=1/32
        end_ep_prob=1/100

        overlay = {}
        rows = buff.split()
        widths = {len(row) for row in rows}
        assert len(widths) == 1
        dim = fn.first(widths)
        assert len(rows) == dim
        
        for y, row in enumerate(rows):
            aps = (codec.get(s, 'white') for s in row)
            overlay.update({(x+ 1, y + 1): ap for x, ap in enumerate(aps)}) 

        super().__init__(dim, start, overlay, slip_prob, end_ep_prob)

class GridworldEnv3(GridworldEnv):
    def __init__(self):
        codec = {'y':'yellow', 'g':'green','r':'red','b':'blue'}
        buff="""y....g..
            ........
            .b.b...r
            .b.b...r
            .b.b....
            .b.b....
            rrrrrr.r
            g.y....."""
        startx = np.random.randint(1, 8)
        starty = np.random.randint(1, 8)
        start=GridWorldState(x=startx, y=starty)
        slip_prob=1/32
        end_ep_prob=1/100

        overlay = {}
        rows = buff.split()
        widths = {len(row) for row in rows}
        assert len(widths) == 1
        dim = fn.first(widths)
        assert len(rows) == dim
        
        for y, row in enumerate(rows):
            aps = (codec.get(s, 'white') for s in row)
            overlay.update({(x+ 1, y + 1): ap for x, ap in enumerate(aps)}) 

        super().__init__(dim, start, overlay, slip_prob, end_ep_prob)

if __name__=='__main__':
    gw = GridworldEnv1()

    action_seq = ['↑', '↓', '←', '→']

    for a in action_seq:
        obs, reward, done, _ = gw.step(a)
        print(obs, done, gw.get_events())



