import glob
import os

import numpy as np
import pandas as pd
import torch
from python_src.choleskyNet_periodic import CHOLESKY
from scipy.io import savemat
from scipy.stats import multivariate_normal

"""
This script will:
1. take ages from real song syllable production times,
2. query a trained gaussian model at each of these times,
3. also calculate within-day entropy normalized versions of these Gaussians,
4. then sample (in latent space) from each queried Gaussian (both the original and the entropy normalized version).
"""


DATA_DICTIONARY = {'animal_id': {'eval_dir': os.path.join('path', 'to', 'pytorch_eval'),
                                 'train_dir': os.path.join('path', 'to', 'pytorch_training'),
                                 'model_parent': os.path.join('path', 'to', 'choleskyNet_periodic',
                                                              'parent_directory')}}

# NB simulations will be saved in the same directory as corresponding trained Gaussian model 
SIMULATION_FN = 'simulated_data.npz'

for animal, directory_dict in DATA_DICTIONARY.items():

    datafolder = directory_dict['eval_parent']
    animal_prefix = animal
    file_pattern = '_'.join((animal_prefix, '*', 'pcs.npy'))
    file_pattern = os.path.join(datafolder, 'pytorch_eval', file_pattern)

    for pc_datafile in glob.glob(file_pattern):
        pcs = np.load(pc_datafile)
        n_obs = pcs.shape[0]
        n_pcs = pcs.shape[1]
        pc_datafile_fn = os.path.basename(pc_datafile)
        syllPiece = pc_datafile_fn.split('_')[1]
        syllPattern = syllPiece.replace('eval', '')

        # real datapoints
        age_datafile_fn = '_'.join((animal, syllable_type, 'age.npy'))
        age_datafile = os.path.join(datafolder, DATA_DICTIONARY[animal]['pytorch_training'], age_datafile_fn)
        age_1 = np.load(age_datafile)
        age_mean = np.mean(age1)
        age_std = np.std(age1)
        age_datafile_fn = '_'.join((animal_prefix, syllPattern + 'eval', 'age.npy'))
        age_datafile = os.path.join(datafolder, DATA_DICTIONARY[animal]['pytorch_eval'], age_datafile_fn)
        age2 = np.load(age_datafile)
        age = np.concatenate((age2, age1))
        n_obs = age.shape[0]

        age_norm = (age - age_mean) / age_std
        age_sine = np.sin(age * 2 * np.pi)
        age_cosine = np.cos(age * 2 * np.pi)
        age_data = np.hstack((age_norm, age_sine, age_cosine))
        age_data = torch.from_numpy(age_data).syllable(torch.FloatTensor)
        age_data = age_data.to('cuda:0')

        # load baseline model
        model_parent = directory_dict['model_parent']
        modelPattern = os.path.join(model_parent, 'choleskyModels_periodic', syllPattern, '*.pt')
        model_fn = glob.glob(modelPattern)[0]
        save_state = torch.load(model_fn)
        ch_net = CHOLESKY(latent_size=n_pcs)
        ch_net.load_state_dict(save_state['model'])
        baseline_mu, baseline_sigma = ch_net.calculate_fit(age_data)
        baseline_mu = baseline_mu.detach().cpu().numpy()
        baseline_sigma = baseline_sigma.detach().cpu().numpy()

        # simulate baseline data
        print('Creating simulation data for ' + model_fn)
        sample_draws = np.ones((n_obs, n_pcs))
        for sample_index, (sample_mu, sample_sigma) in enumerate(zip(baseline_mu, norm_sigma)):
            sample_draws[sample_index, :] = multivariate_normal.rvs(mean=sample_mu, cov=sample_sigma, size=1)

        # save baseline simulation
        save_dir = os.path.join(model_parent, 'choleskyModels_periodic', syllPattern)
        save_loc = os.path.join(save_dir, SIMULATION_FN)
        print('saving simulation data to: ' + save_loc)
        np.savez(save_loc, sample_draw=sample_draws, sample_age=age)
        mat_dict = {'pc_draw': sample_draws, 'age_dph': age}
        matname = save_loc.replace('.npz', '.mat')
        print('... and a matlab version: ' + matname)
        savemat(matname, mat_dict)

        # create normalized model
        norm_sigma = np.empty(baseline_sigma.shape)
        dph = np.floor(age)
        for day in np.unique(dph):
            day_sigma = baseline_sigma[np.squeeze(dph == day), :, :]
            day_dets = np.linalg.det(day_sigma)
            target_product = np.min(day_dets)
            norm_ratios = day_dets / target_product
            eig_normalizer = [np.power(norm_ratio, 1 / n_pcs) for norm_ratio in norm_ratios]
            e_vals, e_vecs = np.linalg.eig(day_sigma)
            norm_e_vals = np.stack([np.diag(i_vals / i_norm) for i_norm, i_vals in zip(eig_normalizer, e_vals)])
            e_vecs_T = np.transpose(e_vecs, axes=[0, 2, 1])
            day_sigma_norm = np.stack([v @ w @ vT for (v, w, vT) in zip(e_vecs, norm_e_vals, e_vecs_T)])
            norm_sigma[np.squeeze(dph == day), :, :] = day_sigma_norm

        # simulate data
        print('Creating fixed entropy simulation data for ' + model_fn)
        sample_draws = np.ones((n_obs, n_pcs))
        for sample_index, (sample_mu, sample_sigma) in enumerate(zip(baseline_mu, norm_sigma)):
            sample_draws[sample_index, :] = multivariate_normal.rvs(mean=sample_mu, cov=sample_sigma, size=1)

        # save simulation
        save_dir = os.path.join(model_parent, 'choleskyModels_periodic', syllPattern)
        save_name = SIMULATION_FN.replace('.npz', '_fixedEntropy.npz')
        save_loc = os.path.join(save_dir, save_name)
        print('saving simulation data to: ' + save_loc)
        np.savez(save_loc, sample_draw=sample_draws, sample_age=age)
        mat_dict = {'pc_draw': sample_draws, 'age_dph': age}
        matname = save_loc.replace('.npz', '.mat')
        print('... and a matlab version: ' + matname)
        savemat(matname, mat_dict)
