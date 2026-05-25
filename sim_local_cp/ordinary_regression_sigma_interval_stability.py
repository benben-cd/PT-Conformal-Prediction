import os
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datasets import datasets
from ordinary_Regmodel import mse_model, LearnerOptimized


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def resolve_mean_model_path(dataset_name, seed):
    candidates = [
        os.path.join("model", f"{dataset_name}_{seed}.pt"),
        f"{dataset_name}_{seed}.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def split_and_scale_like_fitmodel(X, y, seed):
    # 1) D_total -> D_train_all (80%) + D_test (20%)
    # 2) D_train_all -> D_remain (80%) + D_res (20%)
    # 3) D_remain -> D_train (80%) + D_cal (20%)
    x_train_all, x_test, y_train_all, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )
    x_remain, x_res, y_remain, y_res = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.2,
        random_state=seed + 1,
    )
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_remain,
        y_remain,
        test_size=0.2,
        random_state=seed + 2,
    )

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_cal = np.asarray(x_cal)
    y_cal = np.asarray(y_cal)
    x_res = np.asarray(x_res)
    y_res = np.asarray(y_res)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    scaler_x = StandardScaler().fit(x_train)
    x_train = scaler_x.transform(x_train)
    x_cal = scaler_x.transform(x_cal)
    x_res = scaler_x.transform(x_res)
    x_test = scaler_x.transform(x_test)

    mean_abs_y_train = np.mean(np.abs(y_train))
    y_train = (np.squeeze(y_train) / mean_abs_y_train).reshape(-1, 1)
    y_cal = (np.squeeze(y_cal) / mean_abs_y_train).reshape(-1, 1)
    y_res = (np.squeeze(y_res) / mean_abs_y_train).reshape(-1, 1)
    y_test = (np.squeeze(y_test) / mean_abs_y_train).reshape(-1, 1)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_res": x_res,
        "y_res": y_res,
        "x_test": x_test,
        "y_test": y_test,
        "in_shape": x_train.shape[1],
    }


def load_mean_model(model_path, in_shape, hidden_size, dropout):
    model = mse_model(in_shape=in_shape, hidden_size=hidden_size, dropout=dropout)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def train_one_sigma_model(mean_model, x_res, y_res, in_shape, sigma_seed):
    random.seed(sigma_seed)
    np.random.seed(sigma_seed)
    torch.manual_seed(sigma_seed)

    with torch.no_grad():
        y_res_hat = mean_model(
            torch.from_numpy(x_res).float().to(device).requires_grad_(False)
        ).cpu().numpy()

    sigma_target = np.abs(y_res - y_res_hat)

    sigma_net = mse_model(in_shape=in_shape, hidden_size=64, dropout=0.1)
    sigma_learner = LearnerOptimized(
        model=sigma_net,
        optimizer_class=torch.optim.Adam,
        loss_func=nn.MSELoss(),
        device=device,
        optimizer_params={"lr": 5e-4, "weight_decay": 1e-6},
    )
    sigma_learner.fit(x=x_res, y=sigma_target, epochs=1000, batch_size=64)

    return sigma_net


def predict_sigma_positive(sigma_model, x, min_sigma=1e-6):
    sigma_model.eval()
    with torch.no_grad():
        pred = sigma_model(
            torch.from_numpy(x).float().to(device).requires_grad_(False)
        ).cpu().numpy().reshape(-1)

    # mse_model output is unconstrained; enforce positivity at inference.
    sigma = np.abs(pred) + min_sigma
    return sigma


def compute_adaptive_mean_length(mean_model, sigma_model, x_cal, y_cal, x_test, alpha=0.1):
    mean_model.eval()

    with torch.no_grad():
        y_cal_hat = mean_model(
            torch.from_numpy(x_cal).float().to(device).requires_grad_(False)
        ).cpu().numpy().reshape(-1)
        y_test_hat = mean_model(
            torch.from_numpy(x_test).float().to(device).requires_grad_(False)
        ).cpu().numpy().reshape(-1)

    y_cal_flat = y_cal.reshape(-1)
    sigma_cal = predict_sigma_positive(sigma_model, x_cal)
    sigma_test = predict_sigma_positive(sigma_model, x_test)

    score_cal = np.abs(y_cal_flat - y_cal_hat) / sigma_cal

    n_cal = score_cal.shape[0]
    level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    qhat = np.percentile(np.sort(score_cal), level * 100)

    lengths_test = 2.0 * qhat * sigma_test
    return float(np.mean(lengths_test)), float(qhat)


def main():
    os.makedirs("model", exist_ok=True)
    os.makedirs("interval_stability_results", exist_ok=True)

    alpha = 0.1
    num_model_seeds = 5
    num_sigma_runs = 3

    dataset_names = [
        "meps_19",
        "meps_20",
        "meps_21",
        "bike",
        "blog_data",
        "bio",
        "facebook_1",
        "facebook_2",
        "concrete",
        "star",
    ]

    summary_csv = os.path.join("interval_stability_results", "sigma_interval_stability_summary.csv")
    detail_csv = os.path.join("interval_stability_results", "sigma_interval_stability_detail.csv")

    summary_rows = []
    detail_rows = []

    for dataset_name in dataset_names:
        print(f"\n===== Dataset: {dataset_name} =====")

        per_model_seed_stability = []
        per_model_seed_mean_length = []

        X, y = datasets.GetDataset(dataset_name, "./datasets/")

        for model_seed in range(num_model_seeds):
            random.seed(model_seed)
            np.random.seed(model_seed)
            torch.manual_seed(model_seed)

            split_data = split_and_scale_like_fitmodel(X, y, model_seed)
            in_shape = split_data["in_shape"]

            mean_model_path = resolve_mean_model_path(dataset_name, model_seed)
            if mean_model_path is None:
                print(f"[Skip] mean model missing: {dataset_name}_{model_seed}.pt")
                continue

            mean_model = load_mean_model(
                model_path=mean_model_path,
                in_shape=in_shape,
                hidden_size=64,
                dropout=0.1,
            )

            run_mean_lengths = []

            for run_id in range(num_sigma_runs):
                sigma_seed = model_seed * 100 + run_id
                sigma_model = train_one_sigma_model(
                    mean_model=mean_model,
                    x_res=split_data["x_res"],
                    y_res=split_data["y_res"],
                    in_shape=in_shape,
                    sigma_seed=sigma_seed,
                )

                sigma_path = os.path.join(
                    "model",
                    f"sigma_{dataset_name}_{model_seed}_run{run_id}.pt",
                )
                torch.save(sigma_model.state_dict(), sigma_path)

                mean_len, qhat = compute_adaptive_mean_length(
                    mean_model=mean_model,
                    sigma_model=sigma_model,
                    x_cal=split_data["x_cal"],
                    y_cal=split_data["y_cal"],
                    x_test=split_data["x_test"],
                    alpha=alpha,
                )
                run_mean_lengths.append(mean_len)

                detail_rows.append([
                    dataset_name,
                    model_seed,
                    run_id,
                    sigma_seed,
                    mean_len,
                    qhat,
                    sigma_path,
                ])

                print(
                    f"dataset={dataset_name}, model_seed={model_seed}, run={run_id}, "
                    f"mean_length={mean_len:.6f}, qhat={qhat:.6f}"
                )

            interval_stability_seed = float(np.std(run_mean_lengths))
            mean_length_seed = float(np.mean(run_mean_lengths))

            per_model_seed_stability.append(interval_stability_seed)
            per_model_seed_mean_length.append(mean_length_seed)

            print(
                f"model_seed={model_seed}: "
                f"interval_stability(std over 3 sigma)={interval_stability_seed:.6f}, "
                f"mean_length_over_3sigma={mean_length_seed:.6f}"
            )

        if len(per_model_seed_stability) == 0:
            summary_rows.append([
                dataset_name,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0,
            ])
            continue

        stability_mean = float(np.mean(per_model_seed_stability))
        stability_std_across_model_seed = float(np.std(per_model_seed_stability))
        mean_length_mean = float(np.mean(per_model_seed_mean_length))
        mean_length_std_across_model_seed = float(np.std(per_model_seed_mean_length))

        summary_rows.append([
            dataset_name,
            stability_mean,
            stability_std_across_model_seed,
            mean_length_mean,
            mean_length_std_across_model_seed,
            len(per_model_seed_stability),
        ])

        print(
            f"[Summary] dataset={dataset_name}, "
            f"stability_mean={stability_mean:.6f}, "
            f"stability_std_across_model_seed={stability_std_across_model_seed:.6f}, "
            f"mean_length_mean={mean_length_mean:.6f}, "
            f"mean_length_std_across_model_seed={mean_length_std_across_model_seed:.6f}"
        )

    with open(detail_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "model_seed",
            "sigma_run",
            "sigma_seed",
            "test_mean_interval_length",
            "qhat",
            "sigma_model_path",
        ])
        writer.writerows(detail_rows)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "interval_stability_mean_over_model_seed",
            "interval_stability_std_over_model_seed",
            "test_mean_interval_length_mean_over_model_seed",
            "test_mean_interval_length_std_over_model_seed",
            "num_available_model_seed",
        ])
        writer.writerows(summary_rows)

    print(f"\nSaved detail results to: {detail_csv}")
    print(f"Saved summary results to: {summary_csv}")


if __name__ == "__main__":
    main()
