import argparse
import math
import random

from train_student import train_model


def _sample_log_uniform(low: float, high: float) -> float:
    return math.exp(random.uniform(math.log(low), math.log(high)))


def sample_hyperparameters():
    return {
        "lambda_u": _sample_log_uniform(1e-2, 3.0),
        "learning_rate": _sample_log_uniform(1e-4, 3e-3),
        "weight_decay": _sample_log_uniform(1e-6, 1e-3),
        "epochs": random.randint(20, 100),
    }


def run_random_search(n_trials: int):
    results = []
    best_score = None
    best_params = None

    for trial in range(1, n_trials + 1):
        params = sample_hyperparameters()
        params["save_path"] = f"student_trial_{trial}_best.pt"

        score = train_model(params)
        result = {
            "trial": trial,
            "params": params,
            "score": score,
        }
        results.append(result)

        print(f"Trial {trial}/{n_trials}")
        print(f"  params: {params}")
        print(f"  score:  {score:.6f}")

        if best_score is None or score > best_score:
            best_score = score
            best_params = params

    print("\nBest result")
    print(f"  score:  {best_score:.6f}")
    print(f"  params: {best_params}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    n_trials = args.trials
    run_random_search(n_trials=n_trials)


if __name__ == "__main__":
    main()
