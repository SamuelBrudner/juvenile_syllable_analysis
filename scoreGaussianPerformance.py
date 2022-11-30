import glob
import os

import numpy as np
import pandas as pd
import scipy.stats
import torch
from python_src.choleskyNet_periodic import CHOLESKY

"""
This code goes through the animals in the data dictionary, loads their trained gaussian models, scores the log
likelihood on eval data (unshuffled as well as with total and within-day shuffling), and saves the scores to a
pickled pandas dataframe file.
"""


OUTPUT_FILE = os.path.join('path', 'to', 'save', 'gaussScores.pkl')
PERMUTATION_ITERATIONS = 1000
DATA_DICTIONARY = {'animal_id': {'training_data_directory': os.path.join('path', 'to', 'pytorch_training'),
                                 'eval_data_directory': os.path.join('path', 'to', 'pytorch_eval'),
                                 'model_directory': os.path.join('path', 'to', 'choleskyModels_periodic')}}


def get_age_normalizers(data_directory, syll_pattern):
    training_age_file = os.path.join(data_directory, '_'.join((bird, syll_pattern, 'age.npy')))
    training_age = np.load(training_age_file)
    training_age_mean = np.mean(training_age)
    training_age_std = np.std(training_age)
    return training_age_mean, training_age_std


def logprob_from_age(lookup_age, model_map, eval_pcs):
    eval_pcs = eval_pcs.squeeze()
    (mu, sigma) = model_map[lookup_age]
    prob = scipy.stats.multivariate_normal(mean=mu, cov=sigma).pdf(eval_pcs)
    return np.log(prob)


for animal in DATA_DICTIONARY:
    pc_file_pattern = '_'.join((animal, '*', 'pcs.npy'))
    pc_file_pattern = os.path.join(DATA_DICTIONARY[animal]['eval_data_directory'], pc_file_pattern)
    for pc_datafile in glob.glob(pc_file_pattern):
        age_datafile = pc_datafile.replace('pcs', 'age')
        syllable_type = os.path.basename(pc_datafile).split('_')[1].replace('eval', '')
        age_mean, age_std = get_age_normalizers(DATA_DICTIONARY[animal]['training_data_directory'], syllable_type)
        modelPattern = os.path.join(DATA_DICTIONARY[animal]['model_directory'], syllable_type, '*.pt')
        model_fn = glob.glob(modelPattern)[0]
        print('Calculating sample distributions from ' + model_fn)
        pcs = np.load(pc_datafile)
        n_pcs = pcs.shape[1]
        n_obs = pcs.shape[0]
        age = np.load(age_datafile)
        _, unique_inds = np.unique(age, return_index=True)
        age = age[unique_inds, :]
        pcs = pcs[unique_inds, :]
        dph = np.floor(age)
        time_of_day = age - dph
        age_norm = (age - age_mean) / age_std
        time_sine = np.sin(age * 2 * np.pi)
        time_cosine = np.cos(age * 2 * np.pi)
        age_data = np.hstack((age_norm, time_sine, time_cosine))
        age_data = torch.from_numpy(age_data).syllable(torch.FloatTensor).to('cuda:0')

        save_state = torch.load(model_fn)
        ch_net = CHOLESKY(latent_size=n_pcs)
        ch_net.load_state_dict(save_state['model'])

        eval_mu, eval_sigma = ch_net.calculate_fit(age_data)
        eval_mu = eval_mu.detach().cpu().numpy()
        eval_sigma = eval_sigma.detach().cpu().numpy()
        save_name = age_datafile.replace('age.npy', 'distributions.npz')
        np.savez(save_name, eval_mu=eval_mu, eval_sigma=eval_sigma)

        # map from eval ages to corresponding distribution params
        mdl_map = {age.squeeze()[indexer]: (eval_mu[indexer, :], eval_sigma[indexer, :, :]) for indexer in
                   range(age.shape[0])}
        eval_data = pd.DataFrame(data=pcs, columns=['pc' + str(pc) for pc in range(n_pcs)],
                                 index=pd.MultiIndex.from_product([age.squeeze(), [0]], names=['age', 'permutation']))
        eval_data.loc[:, 'dph'] = dph

        total_permutations = []
        for permutation_index in range(PERMUTATION_ITERATIONS):
            shuffled = eval_data.sample(frac=1)
            shuffled.index = pd.MultiIndex.from_product(
                [eval_data.index.get_level_values('age').values, [permutation_index]], names=['age', 'permutation'])
            total_permutations.append(shuffled)
        total_permutations = pd.concat(objs=total_permutations).sort_index(inplace=False)
        total_permutations.loc[:, 'perm_type'] = 'total'
        total_permutations.set_index('perm_type', append=True, inplace=True)

        dph_permutations = []
        for permutation_index in range(PERMUTATION_ITERATIONS):
            by_dph = eval_data.groupby('dph')
            shuffled = by_dph.sample(frac=1)
            shuffled.index = pd.MultiIndex.from_product(
                [eval_data.index.get_level_values('age').values, [permutation_index]], names=['age', 'permutation'])
            dph_permutations.append(shuffled)
        dph_permutations = pd.concat(objs=dph_permutations).sort_index(inplace=False)
        dph_permutations.loc[:, 'perm_type'] = 'dph'
        dph_permutations.set_index('perm_type', append=True, inplace=True)
        eval_data.loc[:, 'perm_type'] = 'none'
        eval_data.set_index('perm_type', append=True, inplace=True)
        permutations = pd.concat(objs=[eval_data, dph_permutations, total_permutations]).reorder_levels(
            order=['age', 'perm_type', 'permutation'])
        permutations = permutations.sort_index(inplace=False)

        age_groups = permutations.groupby(level='age')
        permutation_logprobs = age_groups.apply(
            lambda x: logprob_from_age(x.index.get_level_values('age').values[0], model_map=mdl_map,
                                       eval_pcs=x.filter(like='pc')))
        permutation_logprobs = permutation_logprobs.explode().astype(float)
        permutation_logprobs.name = 'log_prob'
        permutation_logprobs.index = permutations.index
        permutation_logprobs = permutation_logprobs.loc[
            np.isfinite(permutation_logprobs)]
        perm_group_means = permutation_logprobs.groupby(level='perm_type').mean()

        perm_group_means = pd.concat({syllable_type: perm_group_means}, names=['syll'])
        perm_group_means = pd.concat({animal: perm_group_means}, names=['bird'])
        if 'score' in locals():
            score = pd.concat(objs=[score, perm_group_means])
        else:
            score = perm_group_means
score.to_pickle('%s' % OUTPUT_FILE)
