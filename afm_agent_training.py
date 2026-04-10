from env.afm_env import AfmEnvironment

import argparse
import json
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import datetime
from torch.nn import ReLU
from torch import nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

parser = argparse.ArgumentParser()
parser.add_argument("--net_arch", type=int, nargs="+", default=[256, 256], help="Network architecture layer sizes")
parser.add_argument("--num_historic_data", type=int, default=30, help="Number of historic data points")
parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "ppo"], help="RL algorithm to train")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip epsilon")
parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
parser.add_argument("--use_cnn", action="store_true", help="Use CNN feature extractor")
parser.add_argument("--reward_ceiling_offset", type=float, default=10.0, help="Vertical margin above optimal height for crash termination")
parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel training environments")
parser.add_argument("--callback_freq", type=int, default=10000, help="Evaluation and checkpoint frequency (in steps, before division by n_envs)")
parser.add_argument("--reward_ceiling_offset_change_step", type=int, default=None, help="Number of steps after which to change reward_ceiling_offset")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
parser.add_argument("--norm_obs", action=argparse.BooleanOptionalAction, default=True, help="Normalize observations")
parser.add_argument("--norm_reward", action=argparse.BooleanOptionalAction, default=True, help="Normalize rewards")
parser.add_argument("--n_steps", type=int, default=2048, help="PPO only: number of steps per environment per update")
parser.add_argument("--gradient_steps", type=int, default=None, help="SAC only: number of gradient steps per update (default: n_envs)")
parser.add_argument("--tau", type=float, default=0.005, help="SAC only: target network update coefficient")
args = parser.parse_args()

if args.use_cnn and args.num_historic_data < 64:
    raise ValueError(f"CNN requires at least 64 num_historic_data points, got {args.num_historic_data}")

TRAIN_BASE_DIR = "train_results"
train_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

arch_str = "x".join(str(s) for s in args.net_arch)
epsilon_suffix = f"_eps{args.epsilon}" if args.algorithm == "ppo" else ""
cnn_suffix = "_cnn" if args.use_cnn else ""
model_name = f"{args.algorithm}_arch{arch_str}_hist{args.num_historic_data}_rc{args.reward_ceiling_offset}{epsilon_suffix}{cnn_suffix}_{train_date}"
TRAIN_DIR = os.path.join(TRAIN_BASE_DIR, model_name)
os.makedirs(TRAIN_DIR, exist_ok=True)

# Save training configuration
config = vars(args)
with open(os.path.join(TRAIN_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

ENVIRONMENT_DIRS = [
    "environments/pt_111_1a",
    "environments/pt_111_2v",
    "environments/pt_111_2v_1a",
    "environments/pt_111_3v_1a",
]

def make_env_load():
    def _init():
        env = AfmEnvironment(
            surface_configs=[
                {"data_dir_path": env_dir} for env_dir in ENVIRONMENT_DIRS
            ],
            num_historic_data=args.num_historic_data,
            num_actions=1,
            base_reward=10,
            crash_reward=-100.0,
            sigma=2,
            height_offset_reward=0.4,
            reward_exponent=1,
        )
        env = Monitor(env)  #, filename=os.path.join("./logs", f"env_{rank}"))
        return env

    return _init

n_envs = args.n_envs
env_arr = [make_env_load() for _ in range(n_envs)]
vec_env = DummyVecEnv(env_arr)
vec_env = VecNormalize(vec_env, norm_obs=args.norm_obs, norm_reward=args.norm_reward, clip_obs=10.)

eval_env = DummyVecEnv([make_env_load()])
eval_env = VecNormalize(eval_env, norm_obs=args.norm_obs, norm_reward=False, clip_obs=10., training=False)
eval_env.obs_rms = vec_env.obs_rms

class LogConfigCallback(BaseCallback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._logged = False

    def _on_step(self) -> bool:
        if not self._logged:
            config_text = "\n".join(f"    {k}: {v}" for k, v in self.config.items())
            self.logger.record("config", f"```\n{config_text}\n```", exclude="stdout")
            self.logger.dump(self.num_timesteps)
            self._logged = True
        return True


class RewardCeilingOffsetChangeCallback(BaseCallback):
    """Callback to change reward_ceiling_offset after a specified number of steps."""
    def __init__(self, change_step, new_value, verbose=0):
        super().__init__(verbose)
        self.change_step = change_step
        self.new_value = new_value
        self.already_changed = False

    def _on_step(self) -> bool:
        if not self.already_changed and self.num_timesteps >= self.change_step:
            if self.verbose > 0:
                print(f"\nChanging reward_ceiling_offset to {self.new_value} at step {self.num_timesteps}")
            
            # Change in all training environments
            for env in self.model.env.envs:
                env.env.reward_ceiling_offset = self.new_value
            
            self.already_changed = True
        return True


class SyncedEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        if isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = self.model.get_vec_normalize_env().obs_rms

        old_best_reward = self.best_mean_reward
        continue_training = super()._on_step()

        if self.best_mean_reward > old_best_reward:
            vec_env_to_save = self.model.get_vec_normalize_env()
            if vec_env_to_save is not None:
                vec_env_to_save.save(os.path.join(self.best_model_save_path, "best_vecnormalize.pkl"))

        return continue_training

eval_callback = SyncedEvalCallback(
    eval_env,
    best_model_save_path=os.path.join(TRAIN_DIR, "best_model"),
    log_path=os.path.join(TRAIN_DIR, "eval_logs"),
    eval_freq=args.callback_freq // n_envs,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

class AfmCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=162):
        super().__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(2,   32,  kernel_size=8, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,  64,  kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(64,  128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        num_historic_data = observation_space["df"].shape[0]
        with torch.no_grad():
            sample = torch.zeros(1, 2, num_historic_data)
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim - 2),  # reserve 2 dims for x, y
            nn.ReLU(),
        )

    def forward(self, obs):
        x = torch.stack([obs["df"], obs["dz"]], dim=1)  # (batch, 2, T)
        cnn_out = self.linear(self.cnn(x))
        pos = torch.stack([obs["x"][:, 0], obs["y"][:, 0]], dim=1)  # current position only
        return torch.cat([cnn_out, pos], dim=1)

policy_kwargs = dict(
    net_arch=args.net_arch,
    activation_fn=ReLU,
)
if args.use_cnn:
    policy_kwargs["features_extractor_class"] = AfmCnnExtractor
    policy_kwargs["features_extractor_kwargs"] = dict(features_dim=162)

if args.algorithm == "sac":
    gradient_steps = args.gradient_steps if args.gradient_steps is not None else n_envs
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        verbose=args.verbose,
        tensorboard_log="./tb_logs_sac_final",
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_steps=gradient_steps,
        tau=args.tau,
        device="cuda",
    )
else:
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=args.verbose,
        tensorboard_log="./tb_logs_ppo_final",
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        clip_range=args.epsilon,
        ent_coef=0.01,
        device="cuda",
    )

checkpoint_callback = CheckpointCallback(
    save_freq=args.callback_freq // n_envs,
    save_path=os.path.join(TRAIN_DIR, "models"),
    name_prefix=model_name,
    save_replay_buffer=False,
    save_vecnormalize=True,
)

callbacks = [checkpoint_callback, eval_callback, LogConfigCallback(config)]

# Add reward_ceiling_offset change callback if specified
if args.reward_ceiling_offset_change_step is not None:
    reward_ceiling_callback = RewardCeilingOffsetChangeCallback(
        change_step=args.reward_ceiling_offset_change_step,
        new_value=args.reward_ceiling_offset,
        verbose=args.verbose
    )
    callbacks.append(reward_ceiling_callback)

model.learn(
    total_timesteps=10000000,
    log_interval=1,
    progress_bar=False,
    tb_log_name=model_name,#"sac_afm_env" + train_date,
    callback=CallbackList(callbacks),
)
model.save(os.path.join(TRAIN_DIR, "final_model"))
vec_env.save(os.path.join(TRAIN_DIR, "vec_normalize.pkl"))