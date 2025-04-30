from termcolor import colored
from   wandb.apis.public import Run as WandbRun
import wandb, pathlib, pandas as pd, argparse, yaml, functools
from typing import Callable
from matplotlib import rcParams, font_manager as fm

if any("Nimbus Roman" in f.name for f in fm.fontManager.ttflist):
    rcParams["font.family"] = "Nimbus Roman"
else: rcParams["font.family"] = "serif"


PWD     = pathlib.Path(__file__).parent
CFG     = yaml.load(open(PWD / 'config.yaml'), Loader=yaml.FullLoader)
PROJECT = CFG['wandb_project']
ENTITY  = CFG['wandb_entity']
DEBUG   = False
api     = wandb.Api()


def smoothing(metric: str) -> Callable:
    if '/' in metric:
        metric = metric.split('/')[-1]
    if   metric == 'grad_norm':
        return functools.partial(pd.Series.ewm, alpha=0.02, adjust=False)
    elif metric == 'episode_reward':
        return functools.partial(pd.Series.rolling, window=25, min_periods=1)
    else:
        return functools.partial(pd.Series.rolling, window=25, min_periods=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subdir', type=str, default='eval', choices=['eval', 'train'])
    parser.add_argument('--metrics', nargs='+', default=['grad_norm'])
    parser.add_argument('--plot_dir', type=str, default='plots')
    return parser.parse_args()

def find_wandb_ids() -> list[str]:
    return [
        dir.name.split('-')[-1]
        for dir in PWD.glob('outputs/task-*/default/seed-*/wandb/run-*')
    ]

def pull_metric_df(run_id: str,
                   metrics: list[str],
                   entity: str = ENTITY,
                   project: str = PROJECT) -> pd.DataFrame:
    run: WandbRun = api.run(f'{entity}/{project}/{run_id}')
    hist: pd.DataFrame = run.history(keys=metrics, pandas=True, samples=10_000_000 if not DEBUG else 500) # 10M samples, but we only log every 500th step, so this is more than enough
    
    ens   = run.config["ensemble_size"]
    seed  = run.config["seed"]
    task  = run.config.get("task", "unknown-task")

    long = (
        hist
        .melt   (id_vars="_step", value_vars=metrics, var_name="metric", value_name="value")
        .rename(columns={"_step": "step"})         # keep the real env step
        .assign (run_id=run_id, ensemble_size=ens, task=task, seed=seed)
        .dropna (subset=["value"]) # wandb sometimes logs NaNs
        .sort_values(by="step")
    )
    return long

def plot_metric(df: pd.DataFrame, metric: str, _dir: pathlib.Path, subdir: str = 'eval'):
    import seaborn as sns, matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    BIN = 20_000

    sub = (
        df[df.metric == metric].copy()
          .assign(step_bin=lambda d: (d["step"] // BIN) * BIN)
    )

    ens_sizes = sorted(sub["ensemble_size"].unique())
    palette   = dict(zip(ens_sizes, sns.color_palette("tab10", len(ens_sizes))))

    sns.set_context("paper", font_scale=1.3)
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(0.4)
    ax.grid(axis="y", color="black", alpha=0.25, lw=0.4, ls="--")
    ax.grid(axis="x", visible=False)

    # mean +- 1 SD
    agg = (
        sub.groupby(["ensemble_size", "step_bin"], sort=False)["value"]
           .agg(["mean", "std"]).reset_index()
    )
    for ens in ens_sizes:
        blk = agg[agg.ensemble_size == ens].sort_values("step_bin")
        ax.plot(blk.step_bin, blk["mean"], color=palette[ens], lw=1.2, label=f"{ens}")
        ax.fill_between(
            blk.step_bin,
            blk["mean"] - blk["std"].fillna(0),
            blk["mean"] + blk["std"].fillna(0),
            color=palette[ens], alpha=0.18, lw=0,
        )

    task  = sub["task"].iloc[0]
    mraw  = metric.split("/", 1)[1]
    ax.set_title(task, pad=10)
    ax.set_xlabel("Environment step")
    ax.set_ylabel(mraw.replace("_", " ").capitalize())
    ax.legend(
        title="Ensemble size",
        frameon=False,
        fontsize="small",
        title_fontsize="small",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98))

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _:
            f"{x/1_000_000:.2f}M")
    )
    ax.tick_params(axis="x", labelsize=11)
    plt.tight_layout()

    out_f = _dir / f"{subdir}_{task}_{mraw}.png"
    plt.savefig(out_f, dpi=300)
    plt.close()
    return out_f


def main(args):
    wandb_ids = find_wandb_ids()
    print('\n' + '='*50)
    print(colored('Running script with:', 'blue', attrs=['bold']))
    print(colored(f'  entity:    {ENTITY}', 'cyan'))
    print(colored(f'  project:   {PROJECT}', 'cyan'))
    print(colored(f'  subdir:    {args.subdir}', 'cyan'))
    print(colored(f'  metrics:   {args.metrics}', 'cyan'))
    print(colored(f'  wandb_ids: {wandb_ids}', 'cyan'))
    print('='*50 + '\n')

    dfs: list[pd.DataFrame] = []
    metrics = [f'{args.subdir}/{m}' for m in args.metrics]

    for run_id in wandb_ids:
        print(colored(f'Pulling metrics for {run_id}: {", ".join(metrics)}', 'cyan'))
        df      = pull_metric_df(run_id, metrics)
        print(colored(f'  Pulled {len(df)} rows', 'green'))
        dfs.append(df)
    
    df = pd.concat(dfs)
    
    print(colored('Created dataframe with', 'cyan'),
          colored(len(df), 'green'),
          colored('rows, for', 'cyan'),
          colored(len(dfs), 'green'),
          colored('runs.', 'cyan'))

    df_copy = df.copy()

    for metric in metrics:
        print(colored(f'Smoothing {metric} with ', 'cyan'),
              colored(smoothing(metric).func.__name__, 'green'),
              colored(f' and arguments', 'cyan'),
              colored(','.join(f'{k}={v}' for k, v in smoothing(metric).keywords.items()), 'green'),
              colored('...', 'cyan'))

        for rid in df_copy['run_id'].unique():
            idx = (df_copy['run_id'] == rid) & (df_copy['metric'] == metric)
            if not idx.any(): continue
            series = df_copy.loc[idx].sort_values('step')['value']
            smoothed = smoothing(metric)(series).mean()
            df_copy.loc[idx, 'value'] = smoothed.values

    shortest_end = (
        df_copy.groupby("ensemble_size")["step"].max().min()
    )
    print(colored(f"Culling data at step ≤ {int(shortest_end):,}", "yellow"))

    df_copy = df_copy[df_copy["step"] <= shortest_end].copy()

    plot_dir: pathlib.Path = PWD / args.plot_dir
    if not plot_dir.exists():
        print(colored(f'Directory ', 'cyan'),
              colored(plot_dir, 'green'),
              colored(' does not exist. Creating...', 'cyan'))

        plot_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        print(colored(f'Plotting curves for ', 'cyan'),
              colored(metric, 'green'),
              colored('...', 'cyan'))
        name = plot_metric(df_copy, metric, plot_dir, subdir=args.subdir)
        print(colored(f'  Saved plot to ', 'cyan'),
              colored(name, 'green'))



if __name__ == '__main__':
    import sys
    sys.argv[1:] = ['--subdir', 'train', '--metrics', 'grad_norm', 'episode_reward']
    args = parse_args()
    main(args)



