import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sparsemax import Sparsemax


# ============================================================
# ============================================================

def _validate_prior_mask(prior_mask, input_dim):
    if prior_mask is None:
        return None

    if isinstance(prior_mask, torch.Tensor):
        prior_mask = prior_mask.cpu().numpy()

    prior_mask = np.array(prior_mask, dtype=np.float32)

    if prior_mask.shape != (input_dim, input_dim):
        print(f"warning: prior_mask shape {prior_mask.shape} vs input dim {input_dim}; ignoring")
        return None

    if np.any(prior_mask < 0) or np.any(prior_mask > 1):
        print("warning: prior_mask has values outside [0,1]; normalizing")
        prior_mask = np.clip(prior_mask, 0, 1)

    if np.allclose(prior_mask, 0):
        print("warning: prior_mask is all zeros; using default init")
        return None

    return prior_mask


def _init_mask_from_prior(prior_mask, input_dim):
    """Initialise the global static mask M_0 from the prior W_prior.

    Falls back to a uniform random mask when no prior is supplied.
    """
    if prior_mask is None:
        return np.random.rand(input_dim, input_dim).astype(np.float32)
    return np.asarray(prior_mask, dtype=np.float32)


# ============================================================
# ============================================================

def _build_backbone(input_dim, rep_units=200, rep_layers=3, backbone_type="dragonnet"):
    if backbone_type == "wide_dragonnet":
        rep_units = rep_units * 2

    if backbone_type == "tarnet":
        shared_layers = []
        in_dim = input_dim
        for i in range(rep_layers - 1):
            shared_layers.append(nn.Linear(in_dim, rep_units))
            shared_layers.append(nn.ELU())
            in_dim = rep_units
        shared_trunk = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()

        y0_rep_layer = nn.Sequential(
            nn.Linear(rep_units, rep_units),
            nn.ELU()
        )
        y1_rep_layer = nn.Sequential(
            nn.Linear(rep_units, rep_units),
            nn.ELU()
        )
        return shared_trunk, y0_rep_layer, y1_rep_layer

    else:
        layers = []
        in_dim = input_dim
        for i in range(rep_layers):
            layers.append(nn.Linear(in_dim, rep_units))
            layers.append(nn.ELU())
            in_dim = rep_units
        shared_trunk = nn.Sequential(*layers)
        return shared_trunk, None, None


def _build_heads(rep_units, hyp_units=100, hyp_layers=2):
    y0_layers = []
    in_dim = rep_units
    for i in range(hyp_layers):
        y0_layers.append(nn.Linear(in_dim, hyp_units))
        y0_layers.append(nn.ELU())
        in_dim = hyp_units
    y0_layers.append(nn.Linear(hyp_units, 1))

    y1_layers = []
    in_dim = rep_units
    for i in range(hyp_layers):
        y1_layers.append(nn.Linear(in_dim, hyp_units))
        y1_layers.append(nn.ELU())
        in_dim = hyp_units
    y1_layers.append(nn.Linear(hyp_units, 1))

    y0_head = nn.Sequential(*y0_layers)
    y1_head = nn.Sequential(*y1_layers)
    t_head = nn.Linear(rep_units, 1)

    return y0_head, y1_head, t_head


# ============================================================
# ============================================================

class ACIMModel(nn.Module):
    def __init__(self, input_dim,
                 use_sip=True, sip_type='soft',
                 prior_mask=None,
                 backbone_type="dragonnet", rep_units=200, rep_layers=3,
                 hyp_units=100, hyp_layers=2,
                 dynamic_scale_init=0.1):
        super(ACIMModel, self).__init__()
        self.input_dim = input_dim
        self.use_sip = use_sip
        self.sip_type = sip_type
        self.backbone_type = backbone_type

        validated_prior = _validate_prior_mask(prior_mask, input_dim)
        self.register_buffer('prior_mask_ref',
                             torch.tensor(validated_prior,
                                          dtype=torch.float32) if validated_prior is not None else None)

        # ============================================================
        # ============================================================
        if use_sip and sip_type == 'soft':

            # ========================================================
            #
            #
            # m_i = global_mask + dynamic_weight × gate(x)
            # ========================================================

            init_mask = _init_mask_from_prior(validated_prior, input_dim)

            self.global_mask = nn.Parameter(torch.tensor(init_mask, dtype=torch.float32))

            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ELU(),
                nn.Linear(input_dim * 2, input_dim * input_dim)
            )

            self.dynamic_scale = nn.Parameter(torch.tensor(dynamic_scale_init, dtype=torch.float32))

            self.sparsemax = Sparsemax(dim=1)

        # ============================================================
        # ============================================================
        self.rep_layer, self.y0_rep_layer, self.y1_rep_layer = _build_backbone(
            input_dim, rep_units, rep_layers, backbone_type
        )

        effective_rep_units = rep_units * 2 if backbone_type == "wide_dragonnet" else rep_units


        # ============================================================
        # ============================================================
        self.y0_head, self.y1_head, self.t_head = _build_heads(
            effective_rep_units, hyp_units, hyp_layers
        )

        # ============================================================
        # ============================================================
        # self.epsilon = ... (Removed)

    # ---- get_interaction_mask -------------------------------------------
    def get_interaction_mask(self, x=None):
        if self.use_sip and self.sip_type == "soft":
            if x is None:
                temp_sparsemax = Sparsemax(dim=0)
                base = self.global_mask
                return temp_sparsemax(base).detach().cpu().numpy()
            else:
                batch_size = x.shape[0]
                delta_m = self.gate_network(x).view(batch_size, self.input_dim, self.input_dim)
                m_i = self.global_mask + self.dynamic_scale * delta_m
                return self.sparsemax(m_i).detach().cpu().numpy()
        return None
    # -----------------------------------------------------------------

    def forward(self, x):
        if self.use_sip and self.sip_type == 'soft':
            batch_size = x.shape[0]

            delta_m_flat = self.gate_network(x)
            delta_m = delta_m_flat.view(batch_size, self.input_dim, self.input_dim)

            m_i = self.global_mask + self.dynamic_scale * delta_m

            mask = self.sparsemax(m_i)

            x_processed = torch.einsum('bi,bij->bj', x, mask)
        else:
            x_processed = x

        phi_shared = self.rep_layer(x_processed)

        if self.backbone_type == "tarnet":
            phi_y0 = self.y0_rep_layer(phi_shared)
            phi_y1 = self.y1_rep_layer(phi_shared)
            y0_pred = self.y0_head(phi_y0)
            y1_pred = self.y1_head(phi_y1)
            t_logits = self.t_head(phi_shared)
            phi = phi_shared
        else:
            y0_pred = self.y0_head(phi_shared)
            y1_pred = self.y1_head(phi_shared)
            t_logits = self.t_head(phi_shared)
            phi = phi_shared

        concat_pred = torch.cat([y0_pred, y1_pred, t_logits], dim=1)

        return concat_pred, phi

    def get_potential_outcome(self, x, t_val):
        concat_pred, _ = self.forward(x)
        if t_val == 0:
            return concat_pred[:, 0]
        else:
            return concat_pred[:, 1]