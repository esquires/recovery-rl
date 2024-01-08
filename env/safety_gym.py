import gym
import pickle
import tqdm
import numpy as np
import srlnbc.env.config
from srlnbc.env.simple_safety_gym import SimpleEngine

class SafetyGymEnv(SimpleEngine):
    def __init__(self):
        super().__init__(srlnbc.env.config.point_goal_config)

        # recovery rl requires this. This evironment always ends at confog[num_steps]
        # so set max_episode_steps to something higher
        self._max_episode_steps = self.config["num_steps"] + 5

    def step(self, action) -> None:
        prev_obs = self._obs.copy()
        self._obs, reward, done, info = super().step(action)

        cost = info["cost"]
        info = {
            "constraint": cost > 0,
            "reward": reward,
            "state": prev_obs,
            "next_state": self._obs,
            "action": action,
            "success": info.get("goal_met", False),
        }
        return self._obs, reward, done, info

    def reset(self) -> None:
        self._obs = super().reset()
        return self._obs

    def transition_function(
        self, num_transitions: int, task_demos: bool = False, save_rollouts: bool = False
    ):
        # with open('safety_gym.p', 'rb') as f:
        #     rollouts, transitions = pickle.load(f)
        #     if save_rollouts:
        #         return rollouts
        #     else:
        #         return transitions

        self.reset()
        transitions = []
        rollouts = []
        done = False
        i = 0
        while i < num_transitions:

            if (i % (max(num_transitions, 100) // 100)) == 0:
                print(f'transition function: {i}')
            rollouts.append([])
            state = self.reset()

            action = np.clip(np.random.randn(2), -1, 1)
            next_state, rew, done, info = self.step(action)
            constraint = info["constraint"]
            reward = rew
            transitions.append(
                (state, action, constraint, next_state, not constraint))
            rollouts[-1].append(
                (state, action, constraint, next_state, not constraint))
            state = next_state
            if constraint:
                state = self.reset()
            i += 1

        with open('safety_gym.p', 'wb') as f:
            pickle.dump([rollouts, transitions], f)

        if save_rollouts:
            return rollouts
        else:
            return transitions



# def env_creator_safety_gym(cfg: dict[str, Any]) -> RhoWrapper:
#     safety_gym_cfg = import_class(cfg["safety_gym_cfg"])
#     env = SimpleEngine(safety_gym_cfg)
#     env = SafetyGymToGymnasiumWrapper(
#         env, cfg["do_render"], cfg["rho_low"], cfg["rho_high"]
#     )
#     env = RhoWrapper(env)
#     return env
#
