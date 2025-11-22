import numpy as np

from circle_game import BoxDomain, Player
from food_manager import FoodManager
from geometry_utils import get_intersect

class Engine:
    def __init__(self, domain: BoxDomain, player: Player, food_manager: FoodManager):
        self.domain = domain
        self.player = player
        self.food_manager = food_manager

        self.W = float(domain.W)
        self.H = float(domain.H)

        self._collided_last = False

    @property
    def x(self) -> float:
        return self.player.x

    @property
    def y(self) -> float:
        return self.player.y

    @property
    def theta(self) -> float:
        return self.player.theta

    def reset(self, *, start_pose: tuple[float, float, float] | None = None, rng=None) -> None:
        if start_pose is None:
            rng = rng or np.random.default_rng()
            r = float(self.player.radius)
            x = float(rng.uniform(r, self.W - r))
            y = float(rng.uniform(r, self.H - r))
            theta = float(rng.uniform(-np.pi, np.pi))
        else:
            x, y, theta = map(float, start_pose)

        self.player.set_pose(x, y, theta)
        self._collided_last = False

        self.food_manager.create_food(grid_size=40, offset=150.0)

    def step(self, delta_theta: float, step_len: float) -> dict:
        cur_theta = self.player.theta
        cur_dir = np.array([np.cos(cur_theta), np.sin(cur_theta)], dtype=float)

        new_dir = self.player.obtain_new_dir(current_dir=cur_dir, turning_angle=float(delta_theta))
        new_dir = np.asarray(new_dir, dtype=float)
        new_dir /= (np.linalg.norm(new_dir) + 1e-12)

        p_old = (self.player.x, self.player.y)
        p_try = (p_old[0] + step_len * new_dir[0], p_old[1] + step_len * new_dir[1])

        collided = False
        self.player.lastPwasInters = False
        for i in range(4):
            wall_start = [self.player.wall_start_coordinates[0][i], self.player.wall_start_coordinates[1][i]]
            wall_end   = [self.player.wall_end_coordinates[0][i],   self.player.wall_end_coordinates[1][i]]
            inter = get_intersect(p_old, p_try, wall_start, wall_end)
            if inter:
                backoff = 1e-5
                p_new = (inter[0] - new_dir[0]*backoff, inter[1] - new_dir[1]*backoff)
                self.player.lastPwasInters = True
                self.player.lastIntersWallInd = i
                collided = True
                break
        else:
            p_new = p_try

        self.player.trajectory.append(p_new)
        self.player.current_tet = float(np.arctan2(new_dir[1], new_dir[0]))
        eaten_idx = self.food_manager.eat_in_swept_region(p_old, p_new, self.player.radius)
        eaten = len(eaten_idx)

        self._collided_last = collided
        return {"eaten": eaten, "collided": collided}

    def foods_remaining(self) -> int:
        return int(self.food_manager.remaining_count)

def main():
    domain = BoxDomain()
    food_manager = FoodManager(width=domain.width)
    food_manager.create_food(grid_size=40, offset=150.0)

    p1_start = (110, 110)
    player = Player(
        color="red",
        start_point=p1_start,
        radius=100,
        run_dist_min=50.0, 
        run_dist_max=60.0,  
        domain=domain,
    )

    engine = Engine(domain=domain, player=player, food_manager=food_manager)

    from rl_env import FatCircleRLEnv, EnvConfig
    env = FatCircleRLEnv(engine=engine, config=EnvConfig(
        max_steps=2000,
        fixed_step_len=40.0,
        turn_max=0.5,
        step_cost=1e-3,
        random_start=True,
        include_last_turn=True,
        include_time_left=True,
    ))

    obs, info = env.reset(seed=42)
    for _ in range(100):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()