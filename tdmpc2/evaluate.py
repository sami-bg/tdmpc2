import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import hydra
import torch
import imageio
import numpy as np
from pathlib import Path
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from common.logger import Logger
from omegaconf import OmegaConf

torch.backends.cudnn.benchmark = True


def evaluate_from_checkpoint(checkpoint_path: Path):
	cfg_dir = checkpoint_path.parent.parent / '.hydra'
	return hydra.main(config_name='config', config_path=str(cfg_dir))(_evaluate)()


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict): return _evaluate(cfg)

def _evaluate(cfg: OmegaConf):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	eval_results_path = Path(hydra.utils.get_original_cwd()).absolute() / 'evaluate_results.csv'
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint, num_ensembles_keep=cfg.ensemble_keep_count, ensemble_idxs=cfg.ensemble_keep_idxs)
	
	### NOTE: To handle logging to wandb without interfering with the training logs (since we use training cfg)
	cfg.save_agent = False  # NOTE Already saved in training

	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	if not eval_results_path.exists():
		print(f'aggregation,horizon_eval,var_coeff,ensemble_size,seed,task,reward,success', file=open(eval_results_path, 'a'))

	for task_idx, task in enumerate(tasks):
		cfg.task = task
		cfg.wandb_name_suffix = f'eval_{cfg.checkpoint.split("/")[-1]}_{cfg.wandb_name_suffix}'
		# NOTE This is because wandb.init gets called in Logger constructor
		# and we want to have a different wandb run for each task. Otherwise,
		# one run will have the `task` name in the multitask case but contain
		# all tasks.
		logger = Logger(cfg)
		if not cfg.multitask:
			task_idx = None
		ep_rewards, ep_successes = [], []
		for i in range(cfg.eval_episodes):
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done:
				action, stats = agent.act(obs, t0=t==0, task=task_idx)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if cfg.save_video:
					frames.append(env.render())
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if cfg.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		# NOTE This saves to file
		print(f'{cfg.ensemble_aggregation},{cfg.horizon_eval},{cfg.var_coeff},{cfg.ensemble_size},{cfg.seed},{task},'
			f'{ep_rewards:.01f},{ep_successes:.02f}', file=open(eval_results_path, 'a'))

		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))

	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	import sys
	if any('checkpoint=' in arg for arg in sys.argv[1:]):
		checkpoint: str = [
			arg for arg in sys.argv[1:]
			if 'checkpoint=' in arg
		][0].replace('checkpoint=', '')
		evaluate_from_checkpoint(Path(checkpoint))
	else: evaluate()
