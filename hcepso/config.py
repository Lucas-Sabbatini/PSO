"""Centralized configuration for HCEPSO. No magic numbers elsewhere."""

# PSO hyperparameters (Table 2, Sun et al. 2025)
POPULATION   = 10
MAX_ITER     = 200
C1           = 2.0
C2           = 2.0
OMEGA_INI    = 1.2
OMEGA_FIN    = 0.8

# Fitness penalty weights
ALPHA        = 1000   # conflict penalty
LAMBDA_      = 100    # weight constraint violation penalty
DELTA        = 100    # center-of-mass displacement penalty

# Chaotic search
MU           = 4.0    # Logistic Map parameter
CHAOS_STEPS  = 10

# Bin geometry (overridden per dataset)
BIN_W        = 100
BIN_H        = 100
# BIN_Q and BIN_DELTA_MAX are read from each instance

# Mutation probability
P_MUTATION   = 0.3

# Experiment
N_RUNS       = 10
BETA_VALUES  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
