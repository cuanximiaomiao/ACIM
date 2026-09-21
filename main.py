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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader

from config import TrainConfig, set_seed, create_results_dir
from data_loader import (load_ihdp_causal, load_jobs_causal, load_rhc_causal,
                         IHDPDataset, JobsDataset, RHCDataset,
                         TrainFoldPreprocessor, get_prior_mask)
from models import ACIMModel
from trainer import train, evaluate, evaluate_jobs, evaluate_rhc_uplift

DATA_PATHS = {
    "ihdp": ("./data/ihdp/ihdp_npci_1-1000.train.npz",
             "./data/ihdp/ihdp_npci_1-1000.test.npz"),
    "jobs": ("./data/jobs/train.npz",
             "./data/jobs/test.npz"),
    "rhc":  ("./data/rhc.csv", None),
}

TEST_FRACTION = 0.2   # 80/20 train+validation / test split
VALID_FRACTION = 0.2  # validation share of the train+validation portion


def get_args():
    p = argparse.ArgumentParser(description="ACIM reference implementation")
    p.add_argument("--dataset", default="rhc", choices=["rhc", "ihdp", "jobs"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reps", type=int, default=5,
                   help="number of paired repetitions; the paper reports seeds 42-46")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--bs", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--results_dir", default="./results")
    return p.parse_args()


def split_and_preprocess(dataset, paths, cfg, rep_id=0):
    """Draw the split and fit all preprocessing on the training fold only.

    rep_id selects the pre-generated replication of the IHDP/Jobs archives. RHC has a
    single file; its split is drawn from the repetition's seed.

    Returns train/valid/test tensors plus the outcome scaler (None for RHC, whose
    outcome is binary).
    """
    y_scaler = None

    if dataset == "rhc":
        X_raw, T_all, Y_all = load_rhc_causal(paths[0], verbose=True)
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
            (X_tr, T_tv[tr_rel], Y_tv[tr_rel], None),
            (X_va, T_tv[va_rel], Y_tv[va_rel], None),
            (X_te, T_all[test_idx], Y_all[test_idx], None),
        )
        make = lambda X, T, Y, tau: RHCDataset(X, T, Y)
        return folds, make, n_features, y_scaler

    if dataset == "ihdp":
        X_tr_all, T_tr_all, Y_tr_all, tau_tr_all = load_ihdp_causal(
            paths[0], rep_id=rep_id, verbose=True, normalize=False)
        X_te, T_te, Y_te, tau_te = load_ihdp_causal(
            paths[1], rep_id=rep_id, verbose=True, normalize=False)
        n_features = X_tr_all.shape[1]

        n_tr = len(X_tr_all)
        sub = torch.randperm(n_tr).tolist()
        split = int((1.0 - VALID_FRACTION) * n_tr)
        tr_rel, va_rel = sub[:split], sub[split:]

        # Outcome scaler statistics are computed on the training fold.
        y_scaler = StandardScaler()
        y_scaler.fit(Y_tr_all[tr_rel].numpy())
        Y_tr_t = torch.tensor(y_scaler.transform(Y_tr_all.numpy()), dtype=torch.float32)
        Y_te_t = torch.tensor(y_scaler.transform(Y_te.numpy()), dtype=torch.float32)

        folds = (
            (X_tr_all[tr_rel], T_tr_all[tr_rel], Y_tr_t[tr_rel], tau_tr_all[tr_rel]),
            (X_tr_all[va_rel], T_tr_all[va_rel], Y_tr_t[va_rel], tau_tr_all[va_rel]),
            (X_te, T_te, Y_te_t, tau_te),
        )
        make = lambda X, T, Y, tau: IHDPDataset(X, T, Y, tau=tau)
        return folds, make, n_features, y_scaler

    if dataset == "jobs":
        # The Jobs archive holds 10 pre-generated replications.
        jobs_rep = rep_id % 10
        X_tr_all, T_tr_all, Y_tr_all, e_tr_all = load_jobs_causal(
            paths[0], rep_id=jobs_rep, verbose=True, normalize=False)
        X_te, T_te, Y_te, e_te = load_jobs_causal(
            paths[1], rep_id=jobs_rep, verbose=True, normalize=False)
        n_features = X_tr_all.shape[1]

        n_tr = len(X_tr_all)
        sub = torch.randperm(n_tr).tolist()
        split = int((1.0 - VALID_FRACTION) * n_tr)
        tr_rel, va_rel = sub[:split], sub[split:]

        y_scaler = MinMaxScaler()
        y_scaler.fit(Y_tr_all[tr_rel].numpy())
        Y_tr_t = torch.tensor(y_scaler.transform(Y_tr_all.numpy()), dtype=torch.float32)
        Y_te_t = torch.tensor(y_scaler.transform(Y_te.numpy()), dtype=torch.float32)

        folds = (
            (X_tr_all[tr_rel], T_tr_all[tr_rel], Y_tr_t[tr_rel], e_tr_all[tr_rel]),
            (X_tr_all[va_rel], T_tr_all[va_rel], Y_tr_t[va_rel], e_tr_all[va_rel]),
            (X_te, T_te, Y_te_t, e_te),
        )
        make = lambda X, T, Y, e: JobsDataset(X, T, Y, e)
        return folds, make, n_features, y_scaler

    raise ValueError(f"unknown dataset: {dataset}")


def evaluate_test(dataset, model, test_loader, device, y_scaler):
    """Evaluate the trained model on the test loader."""
    if dataset == "ihdp":
        pehe, ate = evaluate(model, test_loader, device, y_scaler=y_scaler)
        return {"test_pehe": pehe, "test_ate": ate}
    if dataset == "jobs":
        att_err, att_hat, att_rct = evaluate_jobs(model, test_loader, device,
                                                  y_scaler=y_scaler)
        return {"test_att_error": att_err, "test_att_estimate": att_hat,
                "test_att_rct": att_rct}
    auuc, qini = evaluate_rhc_uplift(model, test_loader, device)
    return {"test_net_auuc": auuc, "test_net_qini": qini}


def run_single(dataset, seed, rep_id, cfg, results_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    print(f"\n=== {dataset.upper()} | seed {seed} | repetition {rep_id} | device {device} ===")

    folds, make_dataset, n_features, y_scaler = split_and_preprocess(
        dataset, DATA_PATHS[dataset], cfg, rep_id=rep_id)
    (X_tr, T_tr, Y_tr, ex_tr), (X_va, T_va, Y_va, ex_va), (X_te, T_te, Y_te, ex_te) = folds

    train_loader = DataLoader(make_dataset(X_tr, T_tr, Y_tr, ex_tr),
                              batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(make_dataset(X_va, T_va, Y_va, ex_va),
                              batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(make_dataset(X_te, T_te, Y_te, ex_te),
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
                           verbose=True, dataset_type=dataset, y_scaler=y_scaler,
                           weight_decay=cfg.weight_decay,
                           lambda_rank=cfg.lambda_rank,
                           lambda_bal=cfg.lambda_bal,
                           lambda_rlearner=cfg.lambda_rlearner,
                           backbone_type=cfg.backbone_type)
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
    results.update(evaluate_test(dataset, model, test_loader, device, y_scaler))

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
