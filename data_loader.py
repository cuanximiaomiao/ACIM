import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
class IHDPDataset(Dataset):

    def __init__(self, X, T, Y_factual, tau=None, verbose=False):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(T, np.ndarray):
            T = torch.tensor(T, dtype=torch.float32)
        if isinstance(Y_factual, np.ndarray):
            Y_factual = torch.tensor(Y_factual, dtype=torch.float32)
        if tau is not None and isinstance(tau, np.ndarray):
            tau = torch.tensor(tau, dtype=torch.float32)

        if T.dim() == 1:
            T = T.reshape(-1, 1)
        if Y_factual.dim() == 1:
            Y_factual = Y_factual.reshape(-1, 1)
        if tau is not None and tau.dim() == 1:
            tau = tau.reshape(-1, 1)

        n = X.shape[0]
        if T.shape[0] != n or Y_factual.shape[0] != n or (tau is not None and tau.shape[0] != n):
            raise ValueError("CausalDataset : inconsistent lengths")

        self.X = X
        self.T = T
        self.Y_f = Y_factual
        self.tau = tau
        self.verbose = verbose

        if verbose:
            print(f"CausalDataset: X={X.shape}, T={T.shape}, Y_f={Y_factual.shape}, tau={None if tau is None else tau.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        t = self.T[idx]
        y = self.Y_f[idx]
        tau = self.tau[idx] if self.tau is not None else None
        if tau is not None:
            return x, t, y, tau
        else:
            return x, t, y


def load_ihdp_causal(path, rep_id=0, verbose=False, normalize=False):
    try:
        data = np.load(path, allow_pickle=True)
        if verbose:
            print(f"[loader] loaded {path}, keys: {list(data.keys())}")

        x = None
        for key in ['x', 'X', 'features']:
            if key in data:
                x = data[key].astype(np.float32)
                break
        if x is None:
            raise KeyError("npz missing feature array in ('x','X','features')")

        if x.ndim == 3:
            if x.shape[2] > 1 and rep_id < x.shape[2]:
                x = x[:, :, rep_id]
            elif x.shape[1] > 1 and rep_id < x.shape[1]:
                x = x[:, rep_id, :]
            else:
                x = x.reshape(x.shape[0], -1)
        elif x.ndim != 2:
            raise ValueError(f"unsupported feature ndim: {x.ndim}")

        t = None
        for key in ['t', 'T']:
            if key in data:
                t = data[key].astype(np.float32)
                break
        if t is None:
            raise KeyError("npz missing treatment array in ('t','T')")

        if t.ndim == 2 and t.shape[1] > 1 and rep_id < t.shape[1]:
            t = t[:, rep_id]
        elif t.ndim == 3:
            t = t[:, rep_id, 0] if t.shape[2] == 1 else t[:, rep_id]
        t = t.reshape(-1, 1)

        yf = None
        if 'yf' in data:
            yf = data['yf'].astype(np.float32)
            if yf.ndim == 2 and yf.shape[1] > 1:
                yf = yf[:, rep_id]
            elif yf.ndim == 3:
                yf = yf[:, rep_id, 0] if yf.shape[2] == 1 else yf[:, rep_id]
            yf = yf.reshape(-1, 1)

        ycf = None
        if 'ycf' in data:
            ycf = data['ycf'].astype(np.float32)
            if ycf.ndim == 2 and ycf.shape[1] > 1:
                ycf = ycf[:, rep_id]
            elif ycf.ndim == 3:
                ycf = ycf[:, rep_id, 0] if ycf.shape[2] == 1 else ycf[:, rep_id]
            ycf = ycf.reshape(-1, 1)

            tau_true = None

            if 'mu0' in data and 'mu1' in data:
                if verbose:
                    print("[loader] found 'mu0' and 'mu1'; computing tau_true.")
                y0 = data['mu0'].astype(np.float32)
                y1 = data['mu1'].astype(np.float32)

                if y0.ndim > 1:
                    y0 = y0[:, rep_id] if rep_id < y0.shape[1] else y0[:, 0]
                    y1 = y1[:, rep_id] if rep_id < y1.shape[1] else y1[:, 0]
                tau_true = (y1 - y0).reshape(-1, 1)

            elif yf is not None and ycf is not None:
                if verbose:
                    print("[loader] not found 'mu0'/'mu1'; falling back to 'yf' and 'ycf' for tau_true.")
                # tau = Y(1) - Y(0)
                y0 = np.where(t == 1, ycf, yf)
                y1 = np.where(t == 1, yf, ycf)
                tau_true = (y1 - y0).reshape(-1, 1)

            elif 'y0' in data and 'y1' in data:
                if verbose:
                    print("[loader] not found 'mu0'/'mu1' or 'yf'/'ycf'; falling back to 'y0' and 'y1' for tau_true.")
                y0 = data['y0'].astype(np.float32)
                y1 = data['y1'].astype(np.float32)
                if y0.ndim > 1:
                    y0 = y0[:, rep_id] if rep_id < y0.shape[1] else y0[:, 0]
                    y1 = y1[:, rep_id] if rep_id < y1.shape[1] else y1[:, 0]
                tau_true = (y1 - y0).reshape(-1, 1)

        X_tensor = torch.tensor(x, dtype=torch.float32)
        T_tensor = torch.tensor(t, dtype=torch.float32)
        Y_f_tensor = torch.tensor(yf, dtype=torch.float32) if yf is not None else None
        tau_tensor = torch.tensor(tau_true, dtype=torch.float32) if tau_true is not None else None

        if normalize:
            X_mean = X_tensor.mean(dim=0, keepdim=True)
            X_std = X_tensor.std(dim=0, keepdim=True) + 1e-8
            X_tensor = (X_tensor - X_mean) / X_std
            if verbose:
                print(f"[loader] feature standardization applied")

        if verbose:
            print(f"[loader] shapes - X: {X_tensor.shape}, T: {T_tensor.shape}, "
                  f"Y_f: {None if Y_f_tensor is None else Y_f_tensor.shape}, "
                  f"tau: {None if tau_tensor is None else tau_tensor.shape}")

        return X_tensor, T_tensor, Y_f_tensor, tau_tensor

    except Exception as e:
        print("failed to load IHDP:", str(e))
        raise


class JobsDataset(Dataset):
    def __init__(self, X, T, Y_factual, e, verbose=False):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(T, np.ndarray):
            T = torch.tensor(T, dtype=torch.float32)
        if isinstance(Y_factual, np.ndarray):
            Y_factual = torch.tensor(Y_factual, dtype=torch.float32)
        if isinstance(e, np.ndarray):
            e = torch.tensor(e, dtype=torch.float32)

        if T.dim() == 1: T = T.reshape(-1, 1)
        if Y_factual.dim() == 1: Y_factual = Y_factual.reshape(-1, 1)
        if e.dim() == 1: e = e.reshape(-1, 1)

        n = X.shape[0]
        if T.shape[0] != n or Y_factual.shape[0] != n or e.shape[0] != n:
            raise ValueError("JobsDataset : inconsistent lengths")

        self.X = X
        self.T = T
        self.Y_f = Y_factual
        self.e = e
        self.verbose = verbose

        if verbose:
            print(f"JobsDataset: X={X.shape}, T={T.shape}, Y_f={Y_factual.shape}, e={e.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y_f[idx], self.e[idx]

def load_jobs_causal(path, rep_id=0, verbose=False, normalize=False):
    try:
        data = np.load(path, allow_pickle=True)
        if verbose:
            print(f"[Jobs loader] loaded {path}, keys: {list(data.keys())}")

        x = data['x'].astype(np.float32)
        if x.ndim == 3:
            x = x[:, :, rep_id]
        X_tensor = torch.tensor(x, dtype=torch.float32)

        t = data['t'].astype(np.float32)
        if t.ndim == 2:
            t = t[:, rep_id]
        T_tensor = torch.tensor(t.reshape(-1, 1), dtype=torch.float32)

        yf = data['yf'].astype(np.float32)
        if yf.ndim == 2:
            yf = yf[:, rep_id]
        Y_f_tensor = torch.tensor(yf.reshape(-1, 1), dtype=torch.float32)

        e = data['e'].astype(np.float32)
        if e.ndim == 2:
            e = e[:, rep_id]
        E_tensor = torch.tensor(e.reshape(-1, 1), dtype=torch.float32)

        if normalize:
            X_mean = X_tensor.mean(dim=0, keepdim=True)
            X_std = X_tensor.std(dim=0, keepdim=True) + 1e-8
            X_tensor = (X_tensor - X_mean) / X_std
            if verbose:
                print(f"[Jobs loader] feature standardization applied")

        if verbose:
            print(f"[Jobs loader] shapes - X: {X_tensor.shape}, T: {T_tensor.shape}, Y_f: {Y_f_tensor.shape}, e: {E_tensor.shape}")

        return X_tensor, T_tensor, Y_f_tensor, E_tensor

    except Exception as e:
        print("failed to load Jobs:", str(e))
        raise


def get_prior_mask(input_dim, group_ratios=[0.3, 0.4, 0.3]):
    if all(isinstance(r, float) for r in group_ratios):
        boundaries = [0]
        cum = 0
        for r in group_ratios:
            cum += int(round(r * input_dim))
            boundaries.append(cum)
        boundaries[-1] = input_dim
    elif all(isinstance(r, int) for r in group_ratios):
        boundaries = [0] + list(group_ratios) + [input_dim]
    else:
        raise ValueError("group_ratios must be a list of float or int")

    mask = np.zeros((input_dim, input_dim), dtype=np.float32)
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        mask[start:end, start:end] = 1.0
    return mask


def create_dataloaders(dataset, batch_size=128, valid_ratio=0.2, num_workers=0, shuffle=True):
    n = len(dataset)
    indices = torch.randperm(n).tolist() if shuffle else list(range(n))
    train_size = int((1 - valid_ratio) * n)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(Subset(dataset, valid_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader

### ========================================== ###
### ========================================== ###

class RHCDataset(Dataset):
    def __init__(self, X, T, Y_factual, verbose=False):
        if isinstance(X, np.ndarray): X = torch.tensor(X, dtype=torch.float32)
        if isinstance(T, np.ndarray): T = torch.tensor(T, dtype=torch.float32)
        if isinstance(Y_factual, np.ndarray): Y_factual = torch.tensor(Y_factual, dtype=torch.float32)

        if T.dim() == 1: T = T.reshape(-1, 1)
        if Y_factual.dim() == 1: Y_factual = Y_factual.reshape(-1, 1)

        self.X = X
        self.T = T
        self.Y_f = Y_factual
        self.verbose = verbose

        if verbose:
            print(f"RHCDataset: X={X.shape}, T={T.shape}, Y_f={Y_factual.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y_f[idx]

### ========================================== ###
### ========================================== ###

class TrainFoldPreprocessor:
    """Column imputation and standardization statistics, fitted on the training fold.

    fit() is called on the training fold; transform() then applies those statistics to
    the validation and test folds.
    """

    def __init__(self):
        self.fill_values_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X_train):
        X_train = np.asarray(X_train, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            self.fill_values_ = np.nanmean(X_train, axis=0)
        self.fill_values_ = np.where(np.isfinite(self.fill_values_), self.fill_values_, 0.0)
        filled = np.where(np.isnan(X_train), self.fill_values_, X_train)
        self.mean_ = filled.mean(axis=0, keepdims=True)
        self.std_ = filled.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        filled = np.where(np.isnan(X), self.fill_values_, X)
        return ((filled - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)


def load_rhc_causal(path, verbose=False):
    """Load the RHC covariates with a fixed dummy column set.

    One-hot encoding is applied to the whole covariate frame, so the design matrix has
    the same width in every repetition. Missing values are preserved as NaN;
    imputation and standardization are applied by TrainFoldPreprocessor afterwards.
    """
    try:
        csv_path = path if path else 'data/rhc.csv'
        df = pd.read_csv(csv_path)
        if verbose:
            print(f"[RHC loader] loaded {csv_path}, raw shape: {df.shape}")

        t = (df['swang1'] == 'RHC').astype(np.float32).values
        y = (df['death'] == 'Yes').astype(np.float32).values

        drop_cols = [
            'swang1', 'death',
            'sadmdte', 'dschdte',
            'dthdte', 'lstctdte',
            'ptid',
            'dth30', 'surv2md1'
        ]

        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        df_features = df.drop(columns=existing_drop_cols)

        df_features = pd.get_dummies(df_features, drop_first=True)

        # Missing values are preserved here and imputed later by TrainFoldPreprocessor.
        x = df_features.values.astype(np.float64)

        T_tensor = torch.tensor(t.reshape(-1, 1), dtype=torch.float32)
        Y_f_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        if verbose:
            print(f"[RHC loader] design matrix: {x.shape}, "
                  f"T=1 ratio: {t.mean():.3f}, Y=1 ratio: {y.mean():.4f}")
            print("[RHC loader] no imputation or standardization applied; "
                  "use TrainFoldPreprocessor fitted on the training fold")

        return x, T_tensor, Y_f_tensor

    except Exception as e:
        print("failed to load RHC:", str(e))
        raise

### ========================================== ###
### ========================================== ###

class HillstromDataset(Dataset):
    def __init__(self, X, T, Y_factual, verbose=False):
        if isinstance(X, np.ndarray): X = torch.tensor(X, dtype=torch.float32)
        if isinstance(T, np.ndarray): T = torch.tensor(T, dtype=torch.float32)
        if isinstance(Y_factual, np.ndarray): Y_factual = torch.tensor(Y_factual, dtype=torch.float32)

        if T.dim() == 1: T = T.reshape(-1, 1)
        if Y_factual.dim() == 1: Y_factual = Y_factual.reshape(-1, 1)

        self.X = X
        self.T = T
        self.Y_f = Y_factual
        self.verbose = verbose

        if verbose:
            print(f"HillstromDataset: X={X.shape}, T={T.shape}, Y_f={Y_factual.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y_f[idx]


def load_hillstrom_causal(path, outcome='conversion', verbose=False, normalize=True):
    try:
        df = pd.read_csv(path)
        if verbose:
            print(f"[Hillstrom loader] loaded {path}, raw shape: {df.shape}")

        t = (df['segment'] != 'No E-Mail').astype(np.float32).values

        if outcome in ('conversion', 'visit'):
            y = df[outcome].astype(np.float32).values
        elif outcome == 'spend':
            y = df['spend'].astype(np.float32).values
        else:
            raise ValueError(f"unknown outcome: {outcome}; expected conversion/visit/spend")

        drop_cols = ['segment', 'visit', 'conversion', 'spend']
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        df_features = df.drop(columns=existing_drop_cols)

        df_features['history'] = np.log1p(df_features['history'].clip(lower=0))

        df_features = pd.get_dummies(df_features, drop_first=True)

        df_features = df_features.fillna(df_features.mean())

        x = df_features.values.astype(np.float32)

        X_tensor = torch.tensor(x, dtype=torch.float32)
        T_tensor = torch.tensor(t.reshape(-1, 1), dtype=torch.float32)
        Y_f_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        if normalize:
            X_mean = X_tensor.mean(dim=0, keepdim=True)
            X_std = X_tensor.std(dim=0, keepdim=True) + 1e-8
            X_tensor = (X_tensor - X_mean) / X_std
            if verbose:
                print(f"[Hillstrom loader] feature standardization applied")

        if verbose:
            print(f"[Hillstrom loader] shapes - X: {X_tensor.shape}, T: {T_tensor.shape}, "
                  f"Y: {Y_f_tensor.shape}, T=1 ratio: {T_tensor.mean():.3f}, Y=1 ratio: {Y_f_tensor.mean():.4f}")

        return X_tensor, T_tensor, Y_f_tensor

    except Exception as e:
        print("failed to load Hillstrom:", str(e))
        raise

