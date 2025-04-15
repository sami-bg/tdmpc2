import sys
import argparse
from evaluate import evaluate
from contextlib import contextmanager
from itertools import product as cartesian_product

@contextmanager
def set_overrides(checkpoint_path: str, task: str, aggregation: str, horizon_eval: int, var_coeff: float, seed: int):
    hydra_overrides = [
        f'++checkpoint={checkpoint_path}',
        f'++task={task}',
        f'++aggregation={aggregation}',
        f'++horizon_eval={horizon_eval}',
        f'++var_coeff={var_coeff}',
        f'++seed={seed}'
    ]
    try: sys.argv.extend(hydra_overrides) ; yield
    finally: 
        for override in hydra_overrides: sys.argv.remove(override)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",                    required=True,  type=str)
    parser.add_argument("--checkpoints", nargs="+",  required=True,  type=str)
    parser.add_argument("--aggregations", nargs="+", required=False, type=str,    default=["mean"])
    parser.add_argument("--horizon_eval", nargs="+", required=False, type=int,    default=[3, 5, 7, 10, 15])
    parser.add_argument("--var_coeff", nargs="+",    required=False, type=float,  default=[0., 0.001, 0.01, 1., 10.])
    parser.add_argument("--seeds", nargs="+",        required=False, type=int,    default=[1, 2, 3])
    return parser.parse_args()


def main():
    args = parse_args()
    for overrides in cartesian_product(
        args.checkpoints,
        (args.task,),  # NOTE Need to wrap task in a tuple to avoid cartesian product unpacking the str
        args.aggregations,
        args.horizon_eval,
        args.var_coeff,
        args.seeds):
        with set_overrides(*overrides): evaluate()


if __name__ == "__main__": main()

