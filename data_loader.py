import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd

ACTG_COVARIATES = [
    'age', 'wtkg', 'hemo', 'homo', 'drugs', 'oprior',
    'z30', 'preanti', 'race', 'gender', 'str2', 'karnof_hi',
]


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

class UpliftDataset(Dataset):
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
            print(f"UpliftDataset: X={X.shape}, T={T.shape}, Y_f={Y_factual.shape}")

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


def load_actg175_causal(path, verbose=False):
    try:
        csv_path = path if path else 'data/ACTG175.csv'
        df = pd.read_csv(csv_path)
        if verbose:
            print(f"[ACTG loader] loaded {csv_path}, raw shape: {df.shape}")

        missing = [col for col in ACTG_COVARIATES if col not in df.columns]
        if missing:
            raise KeyError(f"missing covariate columns: {missing}")

        x = df[ACTG_COVARIATES].values.astype(np.float64)
        t = df['treat'].astype(np.float32).values
        y = (df['Y'] > 0).astype(np.float32).values

        T_tensor = torch.tensor(t.reshape(-1, 1), dtype=torch.float32)
        Y_f_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        if verbose:
            print(f"[ACTG loader] design matrix: {x.shape}, "
                  f"T=1 ratio: {t.mean():.3f}, Y=1 ratio: {y.mean():.4f}")
            print("[ACTG loader] no imputation or standardization applied; "
                  "use TrainFoldPreprocessor fitted on the training fold")

        return x, T_tensor, Y_f_tensor

    except Exception as e:
        print("failed to load ACTG 175:", str(e))
        raise

### ========================================== ###
### ========================================== ###
