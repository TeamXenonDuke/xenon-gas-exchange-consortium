import numpy as np
hp = np.load("data/FV_healthy_ref/12-11-25_reference_hist_profile.npy")
print(hp.shape)           # (2, 50)
centers = hp[0, :]        # bin centers
probs   = hp[1, :]        # probabilities
