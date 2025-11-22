from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    max_steps: int = 2000
    fixed_step_len: float = 40.0
    turn_max: float = 0.5 
    step_cost: float = 1e-3
    random_start: bool = True
    include_last_turn: bool = True
    include_time_left: bool = True



class _PygameRenderer:
    def __init__(self, W, H, fps=60):
        import pygame
        pygame.init()
        self.pygame = pygame
        self.screen = pygame.display.set_mode((int(W), int(H)))
        self.clock = pygame.time.Clock()
        self.fps = fps

    def draw(self, engine):
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False 
        self.screen.fill("black")

        engine.food_manager.draw_food(self.screen)     

        engine.player.draw(self.screen)                

        pg.display.flip()
        self.clock.tick(self.fps)
        return True



class FatCircleRLEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "FatCircleRLEnv"}

    def __init__(self, engine, config: EnvConfig | None = None):
        super().__init__()
        if config is None:
            config = EnvConfig()
        self.cfg = config
        self.engine = engine


        self._viewer = None
        self._render_fps = 100

        base_obs_dim = 4 + 4 
        extra = int(self.cfg.include_last_turn) + int(self.cfg.include_time_left)
        self._obs_dim = base_obs_dim + extra

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-self.cfg.turn_max], dtype=np.float32),
            high=np.array([+self.cfg.turn_max], dtype=np.float32),
            dtype=np.float32,
        )

        self._t = 0
        self._last_turn = 0.0
        self._rng: Optional[np.random.Generator] = None

    def _ensure_viewer(self):
        if getattr(self, "_viewer", None) is None:
            self._viewer = _PygameRenderer(self.engine.W, self.engine.H, self._render_fps)

    def render(self):
        self._ensure_viewer()
        keep_open = self._viewer.draw(self.engine)
        if not keep_open:
            self.close()

    def close(self):
        import pygame
        pygame.quit()
        self._viewer = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng()

        start_pose: Optional[Tuple[float, float, float]] = None
        if options and "start_pose" in options:
            start_pose = tuple(options["start_pose"]) 
        if self.cfg.random_start and start_pose is None:
            try:
                self.engine.reset(start_pose=None, rng=self._rng)
            except TypeError:
                x = float(self._rng.uniform(0.1 * self.engine.W, 0.9 * self.engine.W))
                y = float(self._rng.uniform(0.1 * self.engine.H, 0.9 * self.engine.H))
                theta = float(self._rng.uniform(-np.pi, np.pi))
                self.engine.reset(start_pose=(x, y, theta))
        else:
            self.engine.reset(start_pose=start_pose, rng=self._rng)

        self._t = 0
        self._last_turn = 0.0

        obs = self._make_obs()
        info = {"foods_remaining": int(self.engine.foods_remaining())}
        return obs, info

    def step(self, action):
        a = float(np.clip(action[0], -self.cfg.turn_max, self.cfg.turn_max))
        res = self.engine.step(delta_theta=a, step_len=self.cfg.fixed_step_len)

        eaten = int(res.get("eaten", 0))
        collided = bool(res.get("collided", False))

        reward = float(eaten) - self.cfg.step_cost

        self._t += 1
        self._last_turn = a

        terminated = False
        if hasattr(self.engine, "foods_remaining") and self.engine.foods_remaining() == 0:
            terminated = True

        truncated = self._t >= self.cfg.max_steps

        obs = self._make_obs()
        info = {
            "eaten": eaten,
            "collided": collided,
            "foods_remaining": int(self.engine.foods_remaining()),
            "t": self._t,
        }
        return obs, reward, terminated, truncated, info

    def _make_obs(self) -> np.ndarray:
        x_n = np.float32(self.engine.x / max(self.engine.W, 1e-6))
        y_n = np.float32(self.engine.y / max(self.engine.H, 1e-6))

        s = np.float32(np.sin(self.engine.theta))
        c = np.float32(np.cos(self.engine.theta))

        dL = np.float32(self.engine.x / max(self.engine.W, 1e-6))
        dR = np.float32((self.engine.W - self.engine.x) / max(self.engine.W, 1e-6))
        dB = np.float32(self.engine.y / max(self.engine.H, 1e-6))
        dT = np.float32((self.engine.H - self.engine.y) / max(self.engine.H, 1e-6))

        vec = [x_n, y_n, s, c, dL, dR, dB, dT]

        if self.cfg.include_last_turn:
            vec.append(np.float32(self._last_turn / max(self.cfg.turn_max, 1e-6)))
        if self.cfg.include_time_left:
            time_left = 1.0 - (self._t / max(self.cfg.max_steps, 1))
            vec.append(np.float32(time_left))

        return np.asarray(vec, dtype=np.float32)
