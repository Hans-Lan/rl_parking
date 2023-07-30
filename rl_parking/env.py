from dataclasses import dataclass
from typing import Optional, Tuple, List
import gym
import gym.spaces
from gym.utils import seeding
import numpy as np
import shapely.geometry as sg
from shapely.affinity import rotate
from ray.rllib.env.env_context import EnvContext
from rl_parking.utils.wrapper import TimeLimit


def angle_fix(angle: float) -> float:
    '''
        Fix angle to [-pi, pi)
    '''
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class ParkingConfig:
    '''
        Configuration for parking environment.
    '''

    # Parking car
    car_length: float = 4.998
    car_width: float = 1.910
    car_wheelbase: float = 2.920
    car_axle_to_back: float = 1.102

    # Parking lot
    mode: str = 'parallel'  # 'parallel' or 'vertical' or 'vertical2'
    lot_margin: float = 1.0
    lot_length: float = None
    lot_width: float = None

    # Simulation
    dt: float = 0.1
    max_step: int = 500

    # Training
    total_iters: int = 100
    jump_start: bool = False

    def __post_init__(self):
        if self.lot_length is None:
            self.lot_length = self.car_length + self.lot_margin
        if self.lot_width is None:
            self.lot_width = self.car_width + self.lot_margin


def create_parking_lot(config: ParkingConfig) -> sg.MultiLineString:
    mode = config.mode
    lot_length = config.lot_length
    lot_width = config.lot_width
    if mode == 'parallel':
        l = lot_length / 2 + 1.0
        w = lot_width / 2 + 0.2
        left_line = sg.LineString([(-12, w), (-l, w)])
        right_line = sg.LineString([(12, w), (l, w)])
        wall = sg.LineString([(-12, 8 + w), (12, 8 + w)])
        lot_boundary = sg.LineString([(-l, w), (-l, -w), (l, -w), (l, w)])
        parking_lot = sg.MultiLineString([left_line, right_line, wall, lot_boundary])
    elif mode == 'vertical':
        l = lot_length / 2
        w = lot_width / 2 + 0.2
        left_line = sg.LineString([(-12, l), (-w, l)])
        right_line = sg.LineString([(12, l), (w, l)])
        wall = sg.LineString([(-12, 6.5 + l), (12, 6.5 + l)])
        lot_boundary = sg.LineString([(-w, l), (-w, -l), (w, -l), (w, l)])
        parking_lot = sg.MultiLineString([left_line, right_line, wall, lot_boundary])
    elif mode == 'vertical2':
        l = lot_length / 2
        w = lot_width / 2 + 0.2
        left_wall = sg.LineString([(-w, -l), (-w, l), (-w - 2, l), (-w - 2, 6.5 + l)])
        right_lines = sg.LineString([(w, l), (w + 5, l), (w + 5, -l), (15, -l)])
        upper_wall = sg.LineString([(-w - 2, 6.5 + l), (15, 6.5 + l)])
        lot_boundary = sg.LineString([(-w, -l), (w, -l), (w, l)])
        parking_lot = sg.MultiLineString([left_wall, right_lines, upper_wall, lot_boundary])
    else:
        raise ValueError(f'Unknown parking mode: {mode}')

    return parking_lot


class ParkingLots(gym.Env):

    def __init__(self, config: ParkingConfig):
        self.config = config

        self.parking_lot = create_parking_lot(config)

        car_l = config.car_length
        car_w = config.car_width
        car_d = config.car_axle_to_back

        self.x_limit = 12
        if config.mode == 'parallel':
            self.init_states_high = np.array([5 + car_l, 4 + car_w / 2, np.pi / 4, 0, 0])
            self.init_states_low = np.array([-5 - car_l, 1 + car_w / 2, -np.pi / 4, 0, 0])
            self.terminal_states = np.array([-car_l / 2 + car_d, 0, 0, 0, 0])
        elif config.mode == 'vertical':
            # self.init_states_high = np.array([5 + car_l, 3 + car_l / 2, np.pi / 6, 0, 0])
            # self.init_states_low = np.array([-5 - car_l, 1 + car_l / 2, -np.pi / 6, 0, 0])
            self.init_states_high = np.array([car_l, 3.5 + car_l / 2, np.pi / 12, 0, 0])
            self.init_states_low = np.array([car_w + 1, 2.5 + car_l / 2, -np.pi / 20, 0, 0])
            self.terminal_states = np.array([0, -car_l / 2 + car_d, np.pi / 2, 0, 0])
        elif config.mode == 'vertical2':
            self.init_states_high = np.array([7.0, 3.0 + car_l / 2, np.pi + 0.1, 0, 0])
            self.init_states_low = np.array([5.0, 2.5 + car_l / 2, np.pi - 0.1, 0, 0])
            self.terminal_states = np.array([0, -car_l / 2 + car_d, np.pi / 2, 0, 0])
            self.x_limit = 14.5
        else:
            raise ValueError(f'Unknown parking mode: {config.mode}')
        self.admissible_errors = np.array([0.1, 0.1, 0.1, 0.4, np.pi / 5])
        if config.mode == 'parallel':
            self.admissible_errors[:2] = 0.2

        # action: [a, d_steer]
        self.action_space = gym.spaces.Box(
            low=np.array([-2, -np.pi / 5], dtype=np.float32),
            high=np.array([2, np.pi / 5], dtype=np.float32), 
        )

        # state: [x, y, theta, v, steer]
        self.state = np.array([0, 0, 0, 0, 0])
        # obs: [x, y, cos_theta, sin_theta, v, steer]
        self.observation_space = gym.spaces.Box(
            low=np.array([-15, -10, -1, -1, -1, -np.pi / 5], dtype=np.float32),
            high=np.array([15, 10, 1, 1, 1, np.pi / 5], dtype=np.float32)
        )
        self.frac = 0.
        self.easy = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def inverse_dyna(self, state, step_count: int):
        L = self.config.car_wheelbase
        dt = self.config.dt
        x, y, theta, v, steer = state
        for _ in range(step_count):
            x = x + v * np.cos(theta) * dt
            y = y + v * np.sin(theta) * dt
            theta = theta + v / L * np.tan(steer) * dt
            v = v
            steer = steer

        # limit v and steer
        return np.array([x, y, theta, v, steer])

    def inverse_dyna_rollout(self, state: np.ndarray, steps: int) -> np.ndarray:
        state = state.copy()
        state[3] = 0.3
        commands = [np.pi / 5, 0, -np.pi / 5, 0]
        max_steps = [80, 120, 80, 130]
        for i in range(4):
            state[-1] = commands[i]
            count = np.clip(steps, 0, max_steps[i])
            state = self.inverse_dyna(state, count)
            steps -= count
            if steps <= 0:
                break
        state[3] = 0.0
        state[-1] = 0.0
        return state

    def step(self, action):
        a, d_steer = action
        L = self.config.car_wheelbase
        dt = self.config.dt
        x, y, theta, v, steer = self.state
        newx = x + v * np.cos(theta) * dt
        newy = y + v * np.sin(theta) * dt
        newtheta = theta + v / L * np.tan(steer) * dt
        newv = v + a * dt
        newsteer = steer + d_steer * dt

        # limit v and steer
        newv = np.clip(newv, -1, 1)
        newsteer = np.clip(newsteer, -np.pi / 5, np.pi / 5)
        self.state = np.array([newx, newy, newtheta, newv, newsteer])

        # step cost
        displacement_cost = np.abs(v) * dt
        control_cost = np.abs(a) + np.abs(d_steer)
        # step_cost = 0.1 * displacement_cost + 0.02 * control_cost
        step_cost = 0.

        # check if the car is parked
        parked = self._parked()
        collision = self._collision()
        out_of_range = self._out_of_range()
        if parked and not collision:
            terminal_reward = 10. - self._terminal_cost()
        elif collision or out_of_range:
            terminal_reward = -10.
        else:
            terminal_reward = 0.

        reward = -step_cost + terminal_reward
        done = parked or collision or out_of_range
        info = {
            "parked": parked,
            "collision": collision,
            "easy": self.easy,
            "frac": self.frac,
        }
        return self._get_obs(), reward, done, info

    def reset(self):
        high = self.init_states_high
        low = self.init_states_low
        if self.config.jump_start:
            # easy = self.np_random.binomial(n=1, p=np.clip(1 - self.frac, 0.1, 1.0))
            easy = 1
            self.easy = easy
            # jump from easy init state
            if easy and self.config.mode == 'parallel':
                self.state = self.jump_start_parallel()
                return self._get_obs()
            elif easy and self.config.mode == 'vertical':
                high, low = self.jump_start_vertical()
            elif easy and self.config.mode == 'vertical2':
                high, low = self.jump_start_vertical2()
            else:
                pass
        self.state = self.np_random.uniform(low=low, high=high)
        while self._collision():
            self.state = self.np_random.uniform(low=low, high=high)
        return self._get_obs()

    def set_jump_start(self, iters: int):
        self.frac = iters / self.config.total_iters
    
    def jump_start_vertical(self) -> Tuple[np.ndarray, np.ndarray]:
        car_l = self.config.car_length
        car_w = self.config.car_width
        car_d = self.config.car_axle_to_back

        # if self.frac < 1:
        #     # first jump
        #     p = np.clip(1 - self.frac, 0.2, 0.8)
        #     mode = self.np_random.choice(3, p=[p, (1-p)*0.9, (1-p)*0.1])
        # else:
        #     # second jump
        #     p = np.clip(self.frac - 1, 0.2, 0.8)
        #     mode = self.np_random.choice(3, p=[(1-p)*0.1, (1-p)*0.9, p])

        # mode = self.np_random.choice(3, p=[0.5, 0.3, 0.2])
        if self.frac < 0.5:
            mode = self.np_random.choice(3, p=[0.7, 0.3, 0.0])
        elif self.frac < 1.0:
            p = self.frac * 0.6
            mode = self.np_random.choice(3, p=[1-p, p, 0.0])
        else:
            mode = self.np_random.choice(3, p=[0.1, 0.6, 0.3])
        # mode = 2
        if mode == 0:
            high = np.array([0.1, -car_l / 2 + car_d + 3.5, np.pi / 2 + 0.1, 0, 0])
            low = np.array([-0.1, -car_l / 2 + car_d + 0.2, np.pi / 2 - 0.1, 0, 0])
        elif mode == 1:
            y_bias = self.np_random.uniform(low=0.0, high=2.5)
            ang_bias = np.pi / 2 - (y_bias - 0.0) / 2.5 * np.pi / 3
            if y_bias < 1.5:
                x_bias_high = 0.
                x_bias_low = 0.
            else:
                x_bias_high = (y_bias - 1.5)
                x_bias_low = 0.4
            # print(y_bias, ang_bias, x_bias_high, x_bias_low)
            # high = np.array([0.5, -car_l / 2 + car_d + high_bias, np.pi / 2 + 0.2, 0, 0])
            # low = np.array([0.2, -car_l / 2 + car_d + 3.5, np.pi / 2 - ang_bias, 0, 0])
            high = np.array([0.5 + x_bias_high, car_l / 2 + y_bias + 0.5, ang_bias + 0.2, 0, 0])
            low = np.array([-0.5 + x_bias_low, car_l / 2 + y_bias, ang_bias - 0.2, 0, 0])
        elif mode == 2:
            # low_bias = self.np_random.uniform(low=2.5, high=3.5)
            # angle = (3.5 - low_bias) * np.pi / 6 + 0.3
            high = np.array([5.5, car_l / 2 + 3.5, np.pi / 10, 0, 0]) 
            low = np.array([-2.5, car_l / 2 + 2.5, 0, 0, 0])
        return high, low
    
    def jump_start_vertical2(self) -> Tuple[np.ndarray, np.ndarray]:
        car_l = self.config.car_length
        car_w = self.config.car_width
        car_d = self.config.car_axle_to_back
        if self.frac < 0.5:
            mode = self.np_random.choice(3, p=[0.7, 0.3, 0.0])
        elif self.frac < 1.0:
            p = self.frac * 0.6
            mode = self.np_random.choice(3, p=[1-p, p, 0.0])
        else:
            p = np.clip((self.frac - 1.0) * 0.4, 0.1, 0.7)
            mode = self.np_random.choice(3, p=[0.2 * (1-p), 0.8 * (1-p), p])
        if mode == 0:
            high = np.array([0.1, -car_l / 2 + car_d + 3.5, np.pi / 2 + 0.1, 0, 0])
            low = np.array([-0.1, -car_l / 2 + car_d + 0.2, np.pi / 2 - 0.1, 0, 0])
        elif mode == 1:
            y_bias = self.np_random.uniform(low=0.0, high=2.5)
            ang_bias = np.pi / 2 - (y_bias - 0.0) / 2.5 * np.pi / 3
            if y_bias < 1.5:
                x_bias_high = 0.
                x_bias_low = 0.
            else:
                x_bias_high = (y_bias - 1.5)
                x_bias_low = 0.5
            high = np.array([0.5 + x_bias_high, car_l / 2 + y_bias + 0.5, ang_bias + 0.2, 0, 0])
            low = np.array([-0.5 + x_bias_low, car_l / 2 + y_bias, ang_bias - 0.2, 0, 0])
        elif mode == 2:
            high = np.array([7.5, car_l / 2 + 3.5, np.pi / 15, 0, 0])
            low = np.array([3.5, car_l / 2 + 2.5, 0.0, 0, 0])
        return high, low

    # def final_reward_vertical(self) -> float:
    #     # for vertical parking only, called when time limit is reached
    #     x, y, theta, *_ = self.state
    #     x_error = np.abs(x - self.terminal_states[0])
    #     y_error = np.abs(y - self.terminal_states[1])
    #     theta_error = np.abs(angle_fix(theta - self.terminal_states[2]))
    #     if x_error < self.config.car_width and theta_error < np.pi / 4:
    #         return np.maximum(10 - (x_error + y_error / 2 + theta_error * 2) * 2, 0)
    #     else:
    #         return 0.

    def jump_start_parallel(self) -> np.ndarray:
        car_l = self.config.car_length
        car_d = self.config.car_axle_to_back
        d = -car_l / 2 + car_d
        x = self.np_random.uniform(low=-0.85, high=-0.8) + d
        y = self.np_random.uniform(low=0.0, high=0.05)
        # theta = self.np_random.uniform(low=-np.pi / 4, high=np.pi / 4)
        init_state = np.array([x, y, 0, 0, 0])
        steps = int(self.frac * 300)
        high = np.maximum(10, steps)
        return self.inverse_dyna_rollout(init_state, self.np_random.randint(low=0, high=high))

    def _get_car_geometry(self) -> sg.Polygon:
        x, y, theta, *_ = self.state
        l = self.config.car_length
        w = self.config.car_width
        d = self.config.car_axle_to_back
        car = sg.Polygon([
            (x - d, y + w / 2),
            (x - d, y - w / 2),
            (x + l - d, y - w / 2),
            (x + l - d, y + w / 2),
        ])
        car = rotate(car, theta, origin=(x, y), use_radians=True)
        return car

    def _collision(self) -> bool:
        car = self._get_car_geometry()
        return self.parking_lot.intersects(car)

    def _out_of_range(self) -> bool:
        x, *_ = self.state
        return x < -self.x_limit or x > self.x_limit
    
    def _parked(self) -> bool:
        state_errors = self.state - self.terminal_states
        state_errors[2] = angle_fix(state_errors[2])
        state_errors = np.abs(state_errors)
        return np.all(state_errors < self.admissible_errors)

    def _terminal_cost(self) -> float:
        state_errors = np.abs(self.state - self.terminal_states)
        e_x = state_errors[0]
        e_y = state_errors[1]
        e_theta = state_errors[2]
        return (e_x + e_y + e_theta) * 5

    def _get_obs(self):
        x, y, theta, v, steer = self.state
        return np.array([x, y, np.cos(theta), np.sin(theta), v, steer], dtype=np.float32)


def env_creator(config: EnvContext):
    env_config = ParkingConfig(
        mode=config['mode'],
        total_iters=config['total_iters'],
        jump_start=config['jump_start'],
    )
    return TimeLimit(ParkingLots(env_config), max_episode_steps=env_config.max_step, penalty=-10.)