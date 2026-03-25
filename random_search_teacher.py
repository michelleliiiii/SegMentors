import math
import random

from train_teacher import train_model


def _sample_log_uniform(low: float, high: float) -> float:
    """
    Sample from a log-uniform distribution between low and high.
    """
    return math.exp(random.uniform(math.log(low), math.log(high)))


def sample_hyperparameters():
    """
    Randomly sample one hyperparameter configuration.
    """
    return {
        "learning_rate": _sample_log_uniform(1e-4, 3e-3),
        "weight_decay": _sample_log_uniform(1e-6, 1e-3),
        "epochs": random.randint(20, 60),
        "batch_size": random.choice([4, 8, 16]),
    }


def run_random_search(n_trials: int):
    """
    Run a very simple random search loop.

    Assumes train_model(params) returns a validation score where higher is better.
    """
    results = []
    best_score = None
    best_params = None

    for trial in range(1, n_trials + 1):
        params = sample_hyperparameters()
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
    n_trials = 10
    run_random_search(n_trials=n_trials)


if __name__ == "__main__":
    main()
