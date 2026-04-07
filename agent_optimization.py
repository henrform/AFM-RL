import os
import argparse
import optuna
import json
import gc
from torch.nn import ReLU

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from optuna.pruners import MedianPruner
from optuna.study import MaxTrialsCallback
from optuna.storages import RDBStorage

from env.afm_env import AfmEnvironment

ENVIRONMENT_DIRS = [
    "environments/pt_111_1a",
    "environments/pt_111_2v",
    "environments/pt_111_2v_1a",
    "environments/pt_111_3v_1a",
]

# Database URL for Distributed Optimization
# Replace with your actual DB URL
DB_URL = "postgresql://user:password@your-db-host.com:5432/optuna_db"
STUDY_NAME = "sac_afm_optimization"

parser = argparse.ArgumentParser()
parser.add_argument("--db_url", type=str, default=DB_URL, help="PostgreSQL connection string for the Optuna study storage")
parser.add_argument(
    "--enqueue",
    action="store_true",
    help="If set, enqueue promising initial parameters before optimization",
)

def make_env(num_historic_data, reward_exponent, rank=0):
    """Utility to create the environment with trial-specific parameters."""
    def _init():
        env = AfmEnvironment(
            surface_configs=[{"data_dir_path": env_dir} for env_dir in ENVIRONMENT_DIRS],
            num_historic_data=num_historic_data,
            num_actions=1,
            df_scale=10.0,
            dz_scale=10.0,
            base_reward=10,
            crash_reward=-100.0,
            sigma=2,
            height_offset_reward=0.4,
            reward_exponent=reward_exponent,
        )
        # Monitor is required to track episode rewards
        env = Monitor(env)
        return env
    return _init

# Callback to report intermediate results to Optuna and prune bad trials
class TrialEvalCallback(EvalCallback):
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, **kwargs):
        super().__init__(eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Report the mean reward to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            self.eval_idx += 1
            # Check if Optuna wants to prune this trial
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return continue_training

def objective(trial):
    """The Optuna objective function executed for each trial."""
    
    # 1. Sample Environment Parameters
    num_historic_data = trial.suggest_categorical("num_historic_data", [20, 40, 80, 150, 200, 250, 300, 350, 400])
    reward_exponent = trial.suggest_float("reward_exponent", 1.0, 2.0)

    # 2. Sample SAC Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    tau = trial.suggest_float("tau", 0.005, 0.05, log=True)
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = [256, 256]
    elif net_arch_choice == "medium":
        net_arch = [512, 512]
    else:
        net_arch = [1024, 1024]

    # 3. Create the Training Environment (Vectorized for performance)
    n_envs = 4 # Maximize GPU usage per worker
    train_env_arr = [make_env(num_historic_data, reward_exponent, i) for i in range(n_envs)]
    vec_train_env = DummyVecEnv(train_env_arr)
    vec_train_env = VecNormalize(vec_train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 4. Create the Evaluation Environment (Single env for reliable testing)
    eval_env = DummyVecEnv([make_env(num_historic_data, reward_exponent, 99)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
    # Sync normalization stats from train to eval
    eval_env.obs_rms = vec_train_env.obs_rms

    # Fixed Architecture
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=ReLU,
    )

    log_dir = "./tb_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize SAC
    model = SAC(
        "MultiInputPolicy",
        vec_train_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        policy_kwargs=policy_kwargs,
        gradient_steps=n_envs,
        verbose=0,
        device="cuda",
        tensorboard_log=log_dir
    )

    trial_dir = f"./train_results/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    with open(os.path.join(trial_dir, "config.json"), "w") as f:
        json.dump(trial.params, f, indent=4)

    # Setup the Pruning Callback (Evaluates every 20,000 steps)
    eval_callback = TrialEvalCallback(
        eval_env, 
        trial, 
        best_model_save_path=os.path.join(trial_dir, "best_model"),
        log_path=None, 
        eval_freq=max(250000 // n_envs, 1) # Adjust for n_envs
    )

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=max(100000 // n_envs, 1), # Spaced out to prevent disk bloat
    #     save_path=os.path.join(trial_dir, "checkpoints"),
    #     name_prefix="sac_afm",
    #     save_replay_buffer=False,
    #     save_vecnormalize=True,
    # )

    callbacks = CallbackList([eval_callback]) #, checkpoint_callback])

    # Train the Model
    try:
        model.learn(
            total_timesteps=7_500_000,
            callback=callbacks,
            tb_log_name=f"trial_{trial.number}"
        )
    except optuna.exceptions.StorageInternalError:
        print("Database connection lost. Check your VPN or Neon status.")
        raise
    except Exception as e:
        print(f"Trial failed due to: {e}")
        raise optuna.exceptions.TrialPruned()

    # Tell Optuna if the trial was killed early
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    # Final evaluation of the fully trained agent
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=50)
    
    model.save(os.path.join(trial_dir, "final_model"))
    vec_train_env.save(os.path.join(trial_dir, "vec_normalize.pkl"))
    
    try:
        model.env.close()
        eval_env.close()
    except Exception as e:
        print(f"Non-fatal error closing envs: {e}")
    
    del model
    del vec_train_env
    del eval_env
    del train_env_arr
    
    gc.collect()

    return mean_reward

if __name__ == "__main__":
    args = parser.parse_args()

    # Ensure the DB URL is set correctly
    print(f"Connecting to Optuna Database at: {args.db_url.split('@')[-1]}")
    
    storage = RDBStorage(
        url=args.db_url,
        heartbeat_interval=60,
        grace_period=120,
        engine_kwargs={
            "pool_size": 2,
            "max_overflow": 2,
            "pool_recycle": 280,
            "pool_pre_ping": True,
        }
    )

    # Create or load the shared study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=0, n_warmup_steps=6, n_min_trials=5),
    )

    max_trials_callback = MaxTrialsCallback(80)
    try:
        current_best = study.best_value
    except ValueError:
        current_best = "None (No completed trials yet)"

    print(f"Starting optimization. Current best value: {current_best}")

    promising_params = {
        "num_historic_data": 40, # Use your actual successful values here
        "reward_exponent": 1.0,
        "learning_rate": 0.0003,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "net_arch": "medium",
    }

    if args.enqueue:
        study.enqueue_trial(promising_params)
    
    study.optimize(objective, callbacks=[max_trials_callback])

    print("Optimization finished!")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")