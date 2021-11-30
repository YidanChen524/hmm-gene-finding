from .config import *
from .helpers import *
from math import isclose
import numpy as np


class hmm:
    def __init__(self, init_probs=init_probs, trans_probs=trans_probs,
                 emission_probs=emission_probs, num_obs_seen=num_obs_seen):
        """initialize hmm with given parameters stored in config"""
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs
        self.num_obs_seen = num_obs_seen  # number of observations trained so far, used to update hmm by proportion
        self.K = self.init_probs.shape[0]  # number of hidden states
        self.D = self.emission_probs.shape[1]  # number of observation states

    def __repr__(self):
        """nicely print the probs of the hmm model"""
        output_format = "init_probs:\n{init}\n\ntrans_probs:\n{trans}\n\nemission_probs:\n{emission}\n\nobs_probs:\n{obs}"
        return output_format.format(
            init=self.init_probs,
            trans=str(self.trans_probs),
            emission=str(self.emission_probs))

    def validate_hmm(self):
        """check if the model is valid"""
        print("\nChecking if HMM is valid...")
        # initial probs must sum to 1
        if not isclose(self.init_probs.sum(), 1):
            print("Fail: init_probs does not sum to 1")
        # each row in trans_probs must sum to 1
        elif sum([not isclose(row.sum(), 1) for row in self.trans_probs]) != 0:
            print("Fail: row(s) in trans_probs do not sum to 1")
        # axis 3 in emission_probs must sum to 1
        elif not np.allclose(self.emission_probs.sum(axis=3), 1):
            print("Fail: element in emission_probs axis 3 does not sum to 1")
        # all number should be between 0 and 1, inclusive
        elif np.any(self.init_probs < 0) & np.any(self.init_probs > 1):
            print("Fail: element in init_probs is not in range 0 and 1")
        elif np.any(self.trans_probs < 0) & np.any(self.trans_probs > 1):
            print("Fail: element in trans_probs is not in range 0 and 1")
        elif np.any(self.emission_probs < 0) & np.any(self.emission_probs > 1):
            print("Fail: element in emission_probs is not in range 0 and 1")
        else:
            print("HMM Validated")

    def update_hmm(self, new_init_probs, new_trans_probs, new_emission_probs,  num_new_obs):
        """update hmm model given new calculated probabilities"""
        # the proportion used to merge new probability
        proportion = num_new_obs / (self.num_obs_seen + num_new_obs)
        def _merge_prob(p1, p2): return p1 * (1 - proportion) + p2 * proportion
        # update the model
        self.num_obs_seen += num_new_obs
        self.init_probs = _merge_prob(self.init_probs, new_init_probs)
        self.trans_probs = _merge_prob(self.trans_probs, new_trans_probs)
        self.emission_probs = _merge_prob(self.emission_probs, new_emission_probs)

    def training_by_counting(self, x: str, z: str):
        """translate by counting when given observations x and path z"""
        # initialize matrices to store counts
        x_indices, z_indices = translate_observations_to_indices(x), translate_annotations_to_states(x, z)
        init, trans, emission = np.ones(self.K), np.ones([self.K, self.K]), np.ones([self.K, self.D, self.D, self.D])
        # training by counting
        init[z_indices[0]] += 1
        trans[z_indices[0], z_indices[1]] += 1
        trans[z_indices[1], z_indices[2]] += 1
        for i in range(2, len(x)):
            trans[z_indices[i - 1], z_indices[i]] += 1
            emission[z_indices[i], x_indices[i-2], x_indices[i-1], x_indices[i]] += 1
        # update the hmm model
        init = init / init.sum()
        trans = trans / trans.sum(axis=1, keepdims=True)
        emission = emission / emission.sum(axis=3, keepdims=True)
        self.update_hmm(init, trans, emission, len(x))

    def predict(self, x: str):
        """return the predicted hidden states for x"""
        x_indices = translate_observations_to_indices(x)
        # initialize a track table to keep track of path
        track_table = np.full([self.K, len(x)], -1)
        # initialize a vector to store one column of the w table
        w_log_column = np.log(self.init_probs)
        # fill out the w column and track table
        for n in range(1, len(x)):
            w_log_column_temp = np.zeros(self.K)
            for k in range(self.K):
                temp = w_log_column + np.log(self.trans_probs[:, k])
                if n > 2:
                    temp += np.log(self.emission_probs[:, x_indices[n-2], x_indices[n-1], x_indices[n]])
                track_table[k, n] = temp.argmax()
                w_log_column_temp[k] = temp.max()
            w_log_column = w_log_column_temp
        # backtracking the track table to get the hidden states
        path = [None] * len(x)
        path[len(x) - 1] = w_log_column.argmax()
        for n in range(len(x) - 2, -1, -1):
            path[n] = track_table[path[n + 1], n+1]
        return translate_states_to_annotations(path)

    def save_model(self):
        """save model params to npz file which would update the initial parameters"""
        np.savez("trained_params.npz", init_probs=self.init_probs, trans_probs=self.trans_probs,
                 emission_probs=self.emission_probs, num_obs_seen=self.num_obs_seen)
