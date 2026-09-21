# ACIM: Adaptive Causal Interaction Model for Uplift Modeling

Reference implementation of the ACIM architecture for individual treatment effect and uplift estimation.

## Repository Structure

```text
.
├── config.py             # Hyperparameter specifications
├── copula_transform.py   # Empirical Copula rank transformation utilities
├── data_loader.py        # Data loading and fold-isolated preprocessing routines
├── models.py             # ACIM neural network architecture with interaction gating
├── trainer.py            # Multi-task training loop, balancing loss, and evaluation metrics
├── synthetic_data.py     # Generative routines for synthetic DAG benchmarks (S1–S4)
├── main.py               # Evaluation entry point
└── requirements.txt      # Python environment dependencies