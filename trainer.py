import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# NumPy 2.0 removed np.trapz in favour of np.trapezoid. Resolve once so the code runs
# on either major version.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ==============================
# 1. MultiTaskLoss
# ==============================
class MultiTaskLoss(nn.Module):
    """Homoscedastic-uncertainty weighting of the outcome and propensity losses.

        L_MTL = sum_k [ (1 / (2 sigma_k^2)) * L_k + log(sigma_k^2) ]

    with s_k = log(sigma_k^2) learnable per task.
    """

    def __init__(self, num_tasks=2):
        super(MultiTaskLoss, self).__init__()
        self.num_tasks = num_tasks
        # log_vars = log(sigma^2)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        # losses: [loss_y, loss_t]
        dtype = losses.dtype
        device = losses.device

        # stds = sigma
        stds = (torch.exp(self.log_vars) ** 0.5).to(device).to(dtype)

        # weight = 1 / (2 * sigma^2)
        coeffs = 1.0 / (2 * (stds ** 2) + 1e-8)

        multi_task_losses = coeffs * losses + self.log_vars

        return multi_task_losses.sum()

    def get_weights(self):
        with torch.no_grad():
            return (torch.exp(self.log_vars) ** 0.5).cpu().numpy()


# ==============================
# 2. BCAUSS Loss Components
# ==============================
def BCAUSS_loss_components(
        pred,
        t_true,
        y_true,
        inputs,
        norm_bal_term=True,
        task_type="regression"
):
    y0_predictions = pred[:, 0:1]
    y1_predictions = pred[:, 1:2]
    t_predictions = pred[:, 2:3]

    if t_true.dim() == 1: t_true = t_true.view(-1, 1)
    if y_true.dim() == 1: y_true = y_true.view(-1, 1)

    t_pred = (t_predictions + 0.001) / 1.002

    if task_type == "regression":
        loss0 = torch.sum((1.0 - t_true) * torch.square(y_true - y0_predictions))
        loss1 = torch.sum(t_true * torch.square(y_true - y1_predictions))
    else:
        bce = nn.BCEWithLogitsLoss(reduction='none')
        loss0 = torch.sum((1.0 - t_true) * bce(y0_predictions, y_true))
        loss1 = torch.sum(t_true * bce(y1_predictions, y_true))

    vanilla_loss = loss0 + loss1


    # --- 2. Balancing Loss (BCAUSS Kernel) ---
    w1 = t_true / t_pred
    w0 = (1 - t_true) / (1 - t_pred)

    ones_to_sum = w1 * inputs
    zeros_to_sum = w0 * inputs

    if norm_bal_term:
        sum_ones_weights = torch.sum(w1, 0) + 1e-8
        sum_zeros_weights = torch.sum(w0, 0) + 1e-8
        ones_mean = torch.sum(ones_to_sum, 0) / sum_ones_weights
        zeros_mean = torch.sum(zeros_to_sum, 0) / sum_zeros_weights
    else:
        ones_mean = torch.sum(ones_to_sum, 0)
        zeros_mean = torch.sum(zeros_to_sum, 0)

    # L_bal = || x_bar^(1) - x_bar^(0) ||_2^2, summed over the covariates
    bal_loss = torch.sum((zeros_mean - ones_mean) ** 2)

    return vanilla_loss, bal_loss


def model_dimension_check(model, loader, device):
    print("\n=== model dimension check ===")
    for batch in loader:
        test_input = batch[0][:1].to(device)
        break
    print(f"test input: {test_input.shape}")
    try:
        output = model(test_input)
        if isinstance(output, tuple):
            pred, rep = output
            print(f"model output (pred): {pred.shape}, representation (rep): {rep.shape}")
        else:
            print(f"model output: {output.shape}")
    except Exception as e:
        print(f"dimension check failed: {str(e)}")
    print("=== end of check ===\n")


### ========================================== ###
### ========================================== ###
### ========================================== ###
### ========================================== ###
def procinis(rank_score, reward, t):
    order = np.argsort(rank_score)[::-1]
    r_sorted = reward[order]
    t_sorted = t[order]

    is_t = t_sorted == 1
    is_good = r_sorted == 1

    cum_t1 = np.cumsum(is_t & is_good)
    cum_t0 = np.cumsum(is_t & ~is_good)
    cum_c1 = np.cumsum(~is_t & is_good)
    cum_c0 = np.cumsum(~is_t & ~is_good)

    tot_t1 = max(cum_t1[-1], 1e-8)
    tot_t0 = max(cum_t0[-1], 1e-8)
    tot_c1 = max(cum_c1[-1], 1e-8)
    tot_c0 = max(cum_c0[-1], 1e-8)

    n = len(reward)
    ks = np.clip(np.round(np.arange(2, 101, 2) / 100.0 * n).astype(int), 1, n)

    x_axis = 0.5 * (cum_t0[ks - 1] / tot_t0 + cum_c1[ks - 1] / tot_c1)
    y_axis = 0.5 * (cum_t1[ks - 1] / tot_t1 + cum_c0[ks - 1] / tot_c0)

    x_axis = np.concatenate([[0.0], x_axis])
    y_axis = np.concatenate([[0.0], y_axis])

    return float(_trapezoid(y_axis, x_axis))


def evaluate_rhc_uplift(model, loader, device):
    model.eval()
    all_cate, all_y, all_t = [], [], []

    with torch.no_grad():
        for batch in loader:
            x, t, yf = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            pred = model(x)
            if isinstance(pred, tuple): pred = pred[0]

            y0_prob = torch.sigmoid(pred[:, 0:1])
            y1_prob = torch.sigmoid(pred[:, 1:2])

            cate = (y0_prob - y1_prob).cpu().numpy().flatten()

            all_cate.extend(cate)
            all_y.extend(yf.cpu().numpy().flatten())
            all_t.extend(t.cpu().numpy().flatten())

    all_cate = np.array(all_cate)
    all_y = np.array(all_y)
    all_t = np.array(all_t)

    y_reward = 1.0 - all_y

    order = np.argsort(all_cate)[::-1]
    t_sorted = all_t[order]
    y_sorted = y_reward[order]

    n_t = np.cumsum(t_sorted)
    n_c = np.cumsum(1 - t_sorted)
    y_t = np.cumsum(y_sorted * t_sorted)
    y_c = np.cumsum(y_sorted * (1 - t_sorted))

    n_t_safe = np.where(n_t == 0, 1e-8, n_t)
    n_c_safe = np.where(n_c == 0, 1e-8, n_c)

    uplift_curve = y_t - y_c * (n_t_safe / n_c_safe)
    qini_curve = y_t - y_c * (n_t[-1] / n_c[-1])

    N = len(all_y)
    uplift_curve_norm = uplift_curve / N
    qini_curve_norm = qini_curve / N

    x_axis = np.arange(1, N + 1) / N
    auuc = _trapezoid(uplift_curve_norm, x_axis)
    qini_area = _trapezoid(qini_curve_norm, x_axis)

    random_uplift_area = uplift_curve_norm[-1] * 0.5
    random_qini_area = qini_curve_norm[-1] * 0.5

    net_auuc = auuc - random_uplift_area
    net_qini = qini_area - random_qini_area
    procinis_area = procinis(all_cate, y_reward, all_t)

    return net_auuc, net_qini, procinis_area


def evaluate_positive_uplift(model, loader, device):
    model.eval()
    all_cate, all_y, all_t = [], [], []

    with torch.no_grad():
        for batch in loader:
            x, t, yf = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            pred = model(x)
            if isinstance(pred, tuple): pred = pred[0]

            y0_prob = torch.sigmoid(pred[:, 0:1])
            y1_prob = torch.sigmoid(pred[:, 1:2])
            cate = (y1_prob - y0_prob).cpu().numpy().flatten()

            all_cate.extend(cate)
            all_y.extend(yf.cpu().numpy().flatten())
            all_t.extend(t.cpu().numpy().flatten())

    all_cate = np.array(all_cate)
    all_y = np.array(all_y)
    all_t = np.array(all_t)

    y_reward = all_y

    order = np.argsort(all_cate)[::-1]
    t_sorted = all_t[order]
    y_sorted = y_reward[order]

    n_t = np.cumsum(t_sorted)
    n_c = np.cumsum(1 - t_sorted)
    y_t = np.cumsum(y_sorted * t_sorted)
    y_c = np.cumsum(y_sorted * (1 - t_sorted))

    n_t_safe = np.where(n_t == 0, 1e-8, n_t)
    n_c_safe = np.where(n_c == 0, 1e-8, n_c)

    uplift_curve = y_t - y_c * (n_t_safe / n_c_safe)
    qini_curve = y_t - y_c * (n_t[-1] / n_c[-1])

    N = len(all_y)
    uplift_curve_norm = uplift_curve / N
    qini_curve_norm = qini_curve / N

    x_axis = np.arange(1, N + 1) / N
    auuc = _trapezoid(uplift_curve_norm, x_axis)
    qini_area = _trapezoid(qini_curve_norm, x_axis)

    random_uplift_area = uplift_curve_norm[-1] * 0.5
    random_qini_area = qini_curve_norm[-1] * 0.5

    net_auuc = auuc - random_uplift_area
    net_qini = qini_area - random_qini_area
    procinis_area = procinis(all_cate, y_reward, all_t)

    return net_auuc, net_qini, procinis_area






# ==============================
# ==============================


def train(model, train_loader, valid_loader, device,
          epochs=100, lr=1e-3,
          patience=40, verbose=True,
          clip_grad_norm=0.8,
          weight_decay=1e-5,
          lambda_rank=0.3,
          lambda_bal=1.0,
          backbone_type="dragonnet",
          lambda_rlearner=1.0,
          eval_uplift=None):
    """Train with early stopping on the validation fold.

    Objective: L_total = L_BCAUSS + lambda_rank * L_rank + lambda_rlearner * L_rlearner,
    with L_BCAUSS = L_MTL + lambda_bal * L_bal. The learning rate is constant.
    """
    if eval_uplift is None:
        eval_uplift = evaluate_rhc_uplift
    task_type = "classification"

    mtl = MultiTaskLoss(num_tasks=2).to(device)

    params = [
        {'params': model.parameters(), 'weight_decay': weight_decay},
        {'params': mtl.parameters(), 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params, lr=lr)

    history = {
        "train_loss": [], "valid_loss": [],
        "train_auuc_history": [], "valid_auuc_history": [],
        "train_qini_history": [], "valid_qini_history": [],
        "train_procinis_history": [], "valid_procinis_history": []
    }

    best_val_metric = -float('inf')
    best_model_state = None
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        mtl.train()

        epoch_train_loss = 0.0

        for batch in train_loader:
            if len(batch) == 4:
                x, t, yf, _ = batch
            else:
                x, t, yf = batch
            x, t, yf = x.to(device), t.to(device), yf.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred, rep = model(x)
            y0_pred, y1_pred, t_logits = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
            t_prob = torch.sigmoid(t_logits)

            pred_bcauss = torch.cat([y0_pred, y1_pred, t_prob], dim=1)

            # L_Y and L_bal (batch-level sums)
            vanilla_loss, bal_loss = BCAUSS_loss_components(pred_bcauss, t.float(), yf, x, task_type=task_type)

            # L_T (batch-level mean)
            t_loss = F.binary_cross_entropy_with_logits(t_logits, t.float())

            supervised_losses = torch.stack([vanilla_loss, t_loss])
            mtl_supervised_loss = mtl(supervised_losses)

            # L_rank: cross-entropy between the within-batch CATE distribution and the
            # within-batch proxy-label distribution
            tau_hat = y1_pred - y0_pred

            pred_log_prob = F.log_softmax(tau_hat.squeeze(), dim=0)

            p_t = torch.clamp(t_prob.squeeze(), min=0.01, max=0.99)
            t_sq = t.squeeze()
            yf_sq = yf.squeeze()
            y_star = yf_sq * (t_sq / p_t - (1.0 - t_sq) / (1.0 - p_t))

            true_prob = F.softmax(y_star, dim=0)

            ranking_loss = -torch.sum(true_prob * pred_log_prob)

            # L_total = L_BCAUSS + lambda_rank * L_rank + lambda_rlearner * L_rlearner
            total_loss = mtl_supervised_loss + lambda_rank * ranking_loss
            if lambda_bal > 0:
                total_loss = total_loss + lambda_bal * bal_loss

            if lambda_rlearner > 0:
                y0_prob_r = torch.sigmoid(y0_pred) if task_type == "classification" else y0_pred
                y1_prob_r = torch.sigmoid(y1_pred) if task_type == "classification" else y1_pred
                m_x = t * y1_prob_r + (1.0 - t) * y0_prob_r
                e_x = torch.clamp(t_prob, min=0.01, max=0.99)
                tau_x = y1_prob_r - y0_prob_r                          # CATE
                r_loss = torch.mean(((yf - m_x) - (t - e_x) * tau_x) ** 2)
                total_loss = total_loss + lambda_rlearner * r_loss

            total_loss.backward()
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            epoch_train_loss += float(total_loss.detach().cpu().item())

        # ==============================
        # Validation
        # ==============================
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                if len(batch) == 4:
                    x, t, yf, _ = batch
                else:
                    x, t, yf = batch
                x, t, yf = x.to(device), t.to(device), yf.to(device)

                pred = model(x)
                if isinstance(pred, tuple): pred = pred[0]

                y0_pred, y1_pred, t_logits = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
                t_prob = torch.sigmoid(t_logits)

                pred_bcauss = torch.cat([y0_pred, y1_pred, t_prob], dim=1)

                v_loss, b_loss = BCAUSS_loss_components(pred_bcauss, t.float(), yf, x, task_type=task_type)

                val_bcauss_metric = v_loss + b_loss

                epoch_val_loss += float(val_bcauss_metric.detach().cpu().item())

        avg_train_loss = epoch_train_loss / max(1, len(train_loader))
        avg_val_loss = epoch_val_loss / max(1, len(valid_loader))

        history["train_loss"].append(avg_train_loss)
        history["valid_loss"].append(avg_val_loss)

        train_auuc, train_qini, train_procinis = eval_uplift(model, train_loader, device)
        valid_auuc, valid_qini, valid_procinis = eval_uplift(model, valid_loader, device)

        history["train_auuc_history"].append(train_auuc)
        history["valid_auuc_history"].append(valid_auuc)
        history["train_qini_history"].append(train_qini)
        history["valid_qini_history"].append(valid_qini)
        history["train_procinis_history"].append(train_procinis)
        history["valid_procinis_history"].append(valid_procinis)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(
                f"Epoch {epoch + 1}/{epochs} | Tr Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val AUUC: {valid_auuc:.4f} | Val Qini: {valid_qini:.4f}")

        # ==============================
        # Checkpoint selection: validation Net Qini. Reads only the validation
        # quantities computed above.
        # ==============================
        if valid_qini > best_val_metric + 1e-6:
            best_val_metric = valid_qini
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
            if verbose:
                print(f"New best model! Val Net Qini: {valid_qini:.4f} (AUUC: {valid_auuc:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} | Best Val Net Qini: {best_val_metric:.4f}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history, model
