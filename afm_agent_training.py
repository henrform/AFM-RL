from env.afm_env import AfmEnvironment

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--net_arch", type=int, nargs="+", default=[256, 256], help="Network architecture layer sizes")
parser.add_argument("--num_historic_data", type=int, default=30, help="Number of historic data points")
parser.add_argument("--df_scale", type=float, default=10.0, help="Divisor for df observations")
parser.add_argument("--dz_scale", type=float, default=10.0, help="Divisor for dz observations")
parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "ppo"], help="RL algorithm to train")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip epsilon")
parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
parser.add_argument("--use_cnn", action="store_true", help="Use CNN feature extractor")
args = parser.parse_args()

if args.use_cnn and args.num_historic_data < 64:
    raise ValueError(f"CNN requires at least 64 num_historic_data points, got {args.num_historic_data}")

TRAIN_BASE_DIR = "train_results"
train_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

arch_str = "x".join(str(s) for s in args.net_arch)
epsilon_suffix = f"_eps{args.epsilon}" if args.algorithm == "ppo" else ""
cnn_suffix = "_cnn" if args.use_cnn else ""
model_name = f"{args.algorithm}_afm_model_arch{arch_str}_hist{args.num_historic_data}_df{args.df_scale}_dz{args.dz_scale}{epsilon_suffix}{cnn_suffix}_{train_date}"
TRAIN_DIR = os.path.join(TRAIN_BASE_DIR, model_name)
os.makedirs(TRAIN_DIR, exist_ok=True)

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
            df_scale=args.df_scale,
            dz_scale=args.dz_scale,
            base_reward=10,
            crash_reward=-100.0,
            sigma=2,
            height_offset_reward=0.4,
            reward_exponent=1,
        )
        env = Monitor(env)  #, filename=os.path.join("./logs", f"env_{rank}"))
        return env

    return _init

n_envs = 4
env_arr = [make_env_load() for _ in range(n_envs)]
vec_env = DummyVecEnv(env_arr)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

eval_env = DummyVecEnv([make_env_load()])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
eval_env.obs_rms = vec_env.obs_rms

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
    eval_freq=max(100_000 // n_envs, 1),
    n_eval_episodes=50,
    deterministic=True,
    render=False,
)

class AfmCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # observation is a dict with df, dz, x, y each of shape (num_historic_data,)
        self.cnn = nn.Sequential(
            nn.Conv1d(4,   32,  kernel_size=8, stride=4),  # → 99
            nn.ReLU(),
            nn.Conv1d(32,  64,  kernel_size=4, stride=2),  # → 48
            nn.ReLU(),
            nn.Conv1d(64,  128, kernel_size=4, stride=2),  # → 23
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=4, stride=2),  # → 10
            nn.ReLU(),
            nn.Flatten(),                                   # → 1280
        )
        # Compute flatten dim using a dummy input
        num_historic_data = observation_space["df"].shape[0]
        with torch.no_grad():
            sample = torch.zeros(1, 4, num_historic_data)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        # Stack dict entries into (batch, channels, timesteps)
        x = torch.stack([obs["df"], obs["dz"], obs["x"], obs["y"]], dim=1)
        return self.linear(self.cnn(x))

policy_kwargs = dict(
    net_arch=args.net_arch,
    activation_fn=ReLU,
)
if args.use_cnn:
    policy_kwargs["features_extractor_class"] = AfmCnnExtractor
    policy_kwargs["features_extractor_kwargs"] = dict(features_dim=160)

if args.algorithm == "sac":
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        verbose=args.verbose,
        tensorboard_log="./tb_logs_sac_final",
        policy_kwargs=policy_kwargs,
        gradient_steps=n_envs,
        device="cuda",
    )
else:
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=args.verbose,
        tensorboard_log="./tensorboard_logs_final_envs",
        policy_kwargs=policy_kwargs,
        clip_range=args.epsilon,
        ent_coef=0.01,
        device="cuda",
    )

checkpoint_callback = CheckpointCallback(
    save_freq=50000 // n_envs,
    save_path=os.path.join(TRAIN_DIR, "models"),
    name_prefix=model_name,
    save_replay_buffer=False,
    save_vecnormalize=True,
)

model.learn(
    total_timesteps=10000000,
    log_interval=1,
    progress_bar=False,
    tb_log_name=model_name,#"sac_afm_env" + train_date,
    callback=CallbackList([checkpoint_callback, eval_callback]),
)
model.save(os.path.join(TRAIN_DIR, "final_model"))
vec_env.save(os.path.join(TRAIN_DIR, "vec_normalize.pkl"))