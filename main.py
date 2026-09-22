"""Single entry point for the ACIM reference implementation.

One run trains ACIM for a single random seed and evaluates on the test split after the
best validation checkpoint has been restored. Imputation and standardization are fitted
on the training fold.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TrainConfig, set_seed, create_results_dir
from data_loader import (load_rhc_causal, load_actg175_causal, UpliftDataset,
                         TrainFoldPreprocessor, get_prior_mask)
from models import ACIMModel
from trainer import train, evaluate_rhc_uplift, evaluate_positive_uplift

DATA_PATHS = {
    "rhc": "./data/rhc.csv",
    "actg175": "./data/ACTG175.csv",
}

UPLIFT_EVAL = {
    "rhc": evaluate_rhc_uplift,
    "actg175": evaluate_positive_uplift,
}

TEST_FRACTION = 0.2   # 80/20 train+validation / test split
VALID_FRACTION = 0.2  # validation share of the train+validation portion


def get_args():
    p = argparse.ArgumentParser(description="ACIM reference implementation")
    p.add_argument("--dataset", default="rhc", choices=["rhc", "actg175"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reps", type=int, default=5,
                   help="number of paired repetitions; the paper reports seeds 42-46")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--bs", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--results_dir", default="./results")
    return p.parse_args()


def split_and_preprocess(dataset, csv_path):
    """Draw the split and fit all preprocessing on the training fold only.

    Returns the train/valid/test folds plus the dataset constructor to use for them.
    """
    if dataset == "rhc":
        X_raw, T_all, Y_all = load_rhc_causal(csv_path, verbose=True)
    else:
        X_raw, T_all, Y_all = load_actg175_causal(csv_path, verbose=True)
    n_features = X_raw.shape[1]

    n_total = len(X_raw)
    all_idx = torch.randperm(n_total).tolist()
    cut = int((1.0 - TEST_FRACTION) * n_total)
    tv_idx, test_idx = all_idx[:cut], all_idx[cut:]

    # Split the train+validation portion into train and validation folds.
    n_tv = len(tv_idx)
    sub = torch.randperm(n_tv).tolist()
    split = int((1.0 - VALID_FRACTION) * n_tv)
    tr_rel, va_rel = sub[:split], sub[split:]

    X_tv = X_raw[tv_idx]
    pre = TrainFoldPreprocessor().fit(X_tv[tr_rel])
    X_tr = pre.transform(X_tv[tr_rel])
    X_va = pre.transform(X_tv[va_rel])
    X_te = pre.transform(X_raw[test_idx])

    T_tv, Y_tv = T_all[tv_idx], Y_all[tv_idx]
    folds = (
        (X_tr, T_tv[tr_rel], Y_tv[tr_rel]),
        (X_va, T_tv[va_rel], Y_tv[va_rel]),
        (X_te, T_all[test_idx], Y_all[test_idx]),
    )
    return folds, UpliftDataset, n_features


def evaluate_test(model, test_loader, device, eval_uplift):
    """Evaluate the trained model on the test loader."""
    auuc, qini, procinis = eval_uplift(model, test_loader, device)
    return {"test_net_auuc": auuc, "test_net_qini": qini, "test_procinis": procinis}


def run_single(dataset, seed, rep_id, cfg, results_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    print(f"\n=== {dataset.upper()} | seed {seed} | repetition {rep_id} | device {device} ===")

    folds, make_dataset, n_features = split_and_preprocess(dataset, DATA_PATHS[dataset])
    (X_tr, T_tr, Y_tr), (X_va, T_va, Y_va), (X_te, T_te, Y_te) = folds

    train_loader = DataLoader(make_dataset(X_tr, T_tr, Y_tr),
                              batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(make_dataset(X_va, T_va, Y_va),
                              batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(make_dataset(X_te, T_te, Y_te),
                             batch_size=cfg.batch_size, shuffle=False)

    # Skeleton prior for the interaction mask.
    prior_mask = get_prior_mask(n_features, [0.3, 0.4, 0.3])
    model = ACIMModel(input_dim=n_features, use_sip=True, sip_type="soft",
                      prior_mask=prior_mask,
                      backbone_type=cfg.backbone_type, rep_units=cfg.rep_units,
                      rep_layers=cfg.rep_layers, hyp_units=cfg.hyp_units,
                      hyp_layers=cfg.hyp_layers,
                      dynamic_scale_init=cfg.dynamic_scale_init).to(device)
    print(f">>> ACIM parameters: {sum(p.numel() for p in model.parameters()):,}")

    start = time.time()
    history, model = train(model, train_loader, valid_loader, device,
                           epochs=cfg.epochs, lr=cfg.lr, patience=cfg.patience,
                           verbose=True,
                           weight_decay=cfg.weight_decay,
                           lambda_rank=cfg.lambda_rank,
                           lambda_bal=cfg.lambda_bal,
                           lambda_rlearner=cfg.lambda_rlearner,
                           backbone_type=cfg.backbone_type,
                           eval_uplift=UPLIFT_EVAL[dataset])
    train_time = time.time() - start

    # Final test evaluation.
    results = {
        "dataset": dataset,
        "seed": seed,
        "rep_id": rep_id,
        "train_time": train_time,
        "hparams": {
            "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
            "patience": cfg.patience, "weight_decay": cfg.weight_decay,
            "lambda_rank": cfg.lambda_rank, "lambda_bal": cfg.lambda_bal,
            "lambda_rlearner": cfg.lambda_rlearner,
        },
        "history": history,
    }
    results.update(evaluate_test(model, test_loader, device, UPLIFT_EVAL[dataset]))

    out_dir = os.path.join(results_dir, dataset, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    metric_line = ", ".join(f"{k}={v:.4f}" for k, v in results.items()
                            if k.startswith("test_") and isinstance(v, float))
    print(f"--- Test evaluation (single run, after best checkpoint) ---\n{metric_line}")
    print(f"Results written to {out_dir}")
    return results


def main():
    args = get_args()
    results_dir = create_results_dir(args.results_dir)
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=args.bs)

    print(f"=== ACIM reference implementation | {args.dataset} ===")
    print(f"Repetitions: {args.reps} (seeds {args.seed}-{args.seed + args.reps - 1})")
    print(f"Frozen configuration: epochs={cfg.epochs}, bs={cfg.batch_size}, "
          f"lr={cfg.lr}, patience={cfg.patience}, weight_decay={cfg.weight_decay}")

    all_results = []
    for rep in range(args.reps):
        seed = args.seed + rep
        try:
            all_results.append(run_single(args.dataset, seed, rep, cfg, results_dir))
        except Exception as exc:
            print(f"[warn] seed {seed} (repetition {rep}) failed: {exc}")

    if not all_results:
        print("no repetition completed")
        return

    keys = sorted({k for r in all_results for k in r if k.startswith("test_")})
    print(f"\n=== {args.dataset.upper()}: mean +/- SD over {len(all_results)} seeds ===")
    for k in keys:
        vals = np.array([r[k] for r in all_results if k in r], dtype=float)
        print(f"  {k:<20} {vals.mean():.5f} +/- {vals.std(ddof=1):.5f}")

    summary_path = os.path.join(results_dir, args.dataset, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"dataset": args.dataset, "seeds": [r["seed"] for r in all_results],
                   "runs": all_results}, f, indent=2, default=float)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
