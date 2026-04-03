from env.afm_env import AfmEnvironment

import argparse
import tqdm
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import os
import datetime
from torch.nn import ReLU

parser = argparse.ArgumentParser()
parser.add_argument("--net_arch", type=int, nargs="+", default=[256, 256], help="Network architecture layer sizes")
parser.add_argument("--num_historic_data", type=int, default=30, help="Number of historic data points")
parser.add_argument("--df_scale", type=float, default=10.0, help="Divisor for df observations")
parser.add_argument("--dz_scale", type=float, default=10.0, help="Divisor for dz observations")
parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "ppo"], help="RL algorithm to train")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip epsilon")
args = parser.parse_args()

TRAIN_BASE_DIR = "train_results"
train_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

arch_str = "x".join(str(s) for s in args.net_arch)
epsilon_suffix = f"_eps{args.epsilon}" if args.algorithm == "ppo" else ""
model_name = f"{args.algorithm}_afm_model_arch{arch_str}_hist{args.num_historic_data}_df{args.df_scale}_dz{args.dz_scale}{epsilon_suffix}_{train_date}"
TRAIN_DIR = os.path.join(TRAIN_BASE_DIR, model_name)
os.makedirs(TRAIN_DIR, exist_ok=True)

ENVIRONMENT_DIRS = [
    "environments/pt_111_1a",
    "environments/pt_111_2v",
    "environments/pt_111_2v_1a",
    "environments/pt_111_3v_1a",
]

def make_env_gpu(rank=0):
    def _init():
        env = AfmEnvironment(
            surface_configs=[{
                'surface_path': "materials/pt_111_small_5row_missing.xyz",
                'params_path': "materials/params_code.ini",
            }],
            num_historic_data=30,
            num_actions=1,
        )
        env = Monitor(env)  #, filename=os.path.join("./logs", f"env_{rank}"))
        return env

    return _init


def make_env_load(rank=0):
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
        )
        env = Monitor(env)  #, filename=os.path.join("./logs", f"env_{rank}"))
        return env

    return _init

n_envs = 4
env_arr = [make_env_load(i) for i in range(n_envs)]
vec_env = DummyVecEnv(env_arr)
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10.)

policy_kwargs = dict(
    net_arch=args.net_arch,
    activation_fn=ReLU,
)

if args.algorithm == "sac":
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_logs_final_envs",
        policy_kwargs=policy_kwargs,
        gradient_steps=n_envs,
        device="cuda",
    )
else:
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_logs_final_envs",
        policy_kwargs=policy_kwargs,
        clip_range=args.epsilon,
        ent_coef=0.01,
        device="cuda",
    )

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=os.path.join(TRAIN_DIR, "models"),
    name_prefix=model_name,
    save_replay_buffer=False,
    save_vecnormalize=True,
)

model.learn(
    total_timesteps=50000000,
    log_interval=1,
    progress_bar=False,
    tb_log_name=model_name,#"sac_afm_env" + train_date,
    callback=[checkpoint_callback],
)
model.save(os.path.join(TRAIN_DIR, "final_model"))
vec_env.save(os.path.join(TRAIN_DIR, "vec_normalize.pkl"))