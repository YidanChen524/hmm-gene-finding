"""
set model parameters for initialization
"""
import numpy as np
import os.path

try:
    # try loading previous trained parameters
    params = np.load(os.path.dirname(os.path.abspath(__file__)) + "/trained_params.npz")
    print("trained_params.npz loaded")
    num_obs_seen = params['num_obs_seen']  # number of observations fed to the hmm model for training
    init_probs = params['init_probs']  # initial probability
    trans_probs = params['trans_probs']  # transition probability
    emission_probs = params['emission_probs']  # emission probability
    print("hmm initialized with set parameters")

except:
    # no previous trained parameters present, initialize with zeros
    print("no previous trained parameters present")
    num_obs_seen = 0
    init_probs = np.zeros(68)
    trans_probs = np.zeros([68, 68])
    emission_probs = np.zeros([68, 4, 4, 4])
    print("hmm initialized with zero probs")
