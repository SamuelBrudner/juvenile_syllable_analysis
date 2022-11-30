import glob
import os

import numpy as np
import pandas as pd
import torch
from python_src.choleskyNet_periodic import CHOLESKY
from scipy.stats import multivariate_normal

# How many minutes count as "nearby" a model sample? Used to determine how much training data was nearby any sample,
# hence, whether the sample represents a reasonable query point. To take an extreme case as an illustration,
# model outputs at nighttime query points are calculable, but not meaningful. (Incidentally, nighttime points are not
# computed here anyway.)
NEARBY_DURATION = 30

# Minutes between samples of the model during robust singing
SAMPLE_DT = 5

# pytorch_training should point to the data folder used in savePytorchMats.m
# model_folder is the parent folder of syllable folders containing model checkpoints
DATA_DICTIONARY = {'animal_id': {'pytorch_training': os.path.join('path', 'to', 'pytorch_training'),
                                 'model_folder': os.path.join('path', 'to', 'saved', 'model')}}# this is MODEL_OUTPUT_SUBDIRECTORY in train file

AGGREGATE_ENTROPY_DATAFILE = os.path.join('path', 'to', 'entropy_data.pkl')

for animal in DATA_DICTIONARY:
    file_pattern = '_'.join((animal, '*', 'pcs.npy'))
    file_pattern = os.path.join(DATA_DICTIONARY[animal]['pytorch_training'], file_pattern)

    for pc_datafile in glob.glob(file_pattern):

        pc_datafile_fn = os.path.basename(pc_datafile)
        syllable_type = pc_datafile_fn.split('_')[1]
        age_datafile_fn = '_'.join((animal, syllable_type, 'age.npy'))
        age_datafile = os.path.join(DATA_DICTIONARY[animal]['pytorch_training'], age_datafile_fn)
        modelPattern = os.path.join(DATA_DICTIONARY[animal]['model_folder'], syllable_type, '*.pt')
        model_fn = glob.glob(modelPattern)[0]

        print('Creating timecourse data for ' + model_fn)

        # construct sample ages for timecourse based on real datapoints
        min_in_days = 1 / (24 * 60)
        pcs = np.load(pc_datafile)
        n_pcs = pcs.shape[1]
        age = np.load(age_datafile)
        dph = np.floor(age)
        true_days = np.unique(dph)
        n_true_days = true_days.shape[0]
        time_of_day = age - dph
        min_time = min(time_of_day) - min_in_days
        max_time = max(time_of_day) + min_in_days
        n_daily_samples = np.floor((max_time - min_time) / (SAMPLE_DT * min_in_days)).astype(
            'int').squeeze()
        query_within_day_times = np.linspace(min_time, max_time, n_daily_samples)
        query_ages_2d = np.meshgrid(true_days, query_within_day_times)
        query_ages = query_ages_2d[0] + query_ages_2d[1]
        n_samples = n_true_days * n_daily_samples
        query_ages = query_ages.reshape(n_samples, 1)
        query_ages.sort(axis=0)
        query_dph = np.floor(query_ages)
        query_time_of_day = query_ages - query_dph
        age_mean = np.mean(age)
        age_std = np.std(age)
        query_ages_norm = (query_ages - age_mean) / age_std
        query_sine = np.sin(query_ages * 2 * np.pi)
        query_cosine = np.cos(query_ages * 2 * np.pi)
        query_age_data = np.hstack((query_ages_norm, query_sine, query_cosine))
        query_age_data = torch.from_numpy(query_age_data).syllable(torch.FloatTensor)
        query_age_data = query_age_data.to('cuda:0')

        # Get model distribution features at query timepoints
        save_state = torch.load(model_fn)
        ch_net = CHOLESKY(latent_size=n_pcs)
        ch_net.load_state_dict(save_state['model'])
        query_mu, query_sigma = ch_net.calculate_fit(query_age_data)
        query_mu = query_mu.detach().cpu().numpy()
        query_sigma = query_sigma.detach().cpu().numpy()
        query_mvns = [multivariate_normal(mean=mu, cov=sigma) for mu, sigma in zip(query_mu, query_sigma)]
        query_entropy = np.array([mvn.entropy() for mvn in query_mvns])
        query_entropy = query_entropy.reshape(query_entropy.shape[0], 1)

        # Assess how many training datapoints were nearby each query time
        nearby_duration_hrs = NEARBY_DURATION / 60
        nearby_duration_days = nearby_duration_hrs / 24
        grab_half_duration = nearby_duration_days / 2
        query_n_nearby = np.zeros(query_entropy.shape)
        for query_ind, query_age in enumerate(query_ages):
            grab_interval = np.array([query_age - grab_half_duration, query_age + grab_half_duration])
            grab_indices = (age >= grab_interval[0]) & (age <= grab_interval[1])
            query_n_nearby[query_ind] = np.sum(grab_indices)

        # Save sampled model timecourse
        save_dir = os.path.join(DATA_DICTIONARY[animal]['model_folder'], syllable_type)
        save_loc = os.path.join(save_dir, 'model_timecourse.npz')
        print('saving model time course to ' + save_loc)
        np.savez(save_loc, query_age=query_ages, query_mu=query_mu, query_sigma=query_sigma,
                 query_entropy=query_entropy, query_n_nearby=query_n_nearby, animal=animal, syll=syllable_type)
        save_data = np.hstack((query_ages, query_entropy, query_n_nearby))
        query_df = pd.DataFrame(save_data, columns=['age', 'entropy', 'n_nearby'])
        query_df.loc[:, 'dph'] = np.floor(query_df.loc[:, 'age'])
        query_df.loc[:, 'time_of_day'] = query_df.loc[:, 'age'] - query_df.loc[:, 'dph']
        query_df.loc[:, 'bird'] = animal
        query_df.loc[:, 'syll'] = syllable_type
        if 'entropy_data' in locals():
            entropy_data = pd.concat((entropy_data, query_df))
        else:
            entropy_data = query_df

entropy_data.to_pickle(AGGREGATE_ENTROPY_DATAFILE)

