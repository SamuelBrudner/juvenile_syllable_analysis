import numpy as np
import pandas as pd
import os
import glob

"""
Go through animals and assess the performance (by MSE) of the predicted age network on test data. Also create scores
following shuffling (total shuffle and within-day shuffle).
"""

N_PERMUTATIONS = 1000

for animal in ['list', 'of', 'animal', 'IDs']:
    print(animal)
    datafile = os.path.join('predicted_age_tables', animal + '_predicted_age_table.txt')
    data = pd.read_csv(datafile, sep='\t')
    data = data[data['age'] >= 60]
    data = data[data['age'] < 95]
    data = data[data['partition'] == 'Test']
    for syll in np.unique(data.loc[:, 'type'].values):

        syll_data = data[data['type'] == syll]
        min_pred_age = np.floor(np.min(syll_data['ffnn_predicted_age']))
        max_pred_age = np.ceil(np.max(syll_data['ffnn_predicted_age']))

        syll_data.loc[:, 'dph'] = np.floor(syll_data.loc[:, 'age'])

        syll_data.loc[:, 'sq_err'] = np.square(syll_data.loc[:, 'ffnn_predicted_age'] - syll_data.loc[:, 'age'])
        no_permutation = pd.DataFrame(columns=['sq_err', 'perm_type'])
        no_permutation.loc[0] = [np.mean(syll_data.loc[:, 'sq_err'].values), 'none']

        n_permutations = N_PERMUTATIONS

        total_permutation_mean_sq_err = []
        for permutation_index in range(n_permutations):
            shuffled_age = syll_data.loc[:, 'age'].sample(frac=1)
            total_permutation_mean_sq_err.append(
                np.mean(np.square(syll_data.loc[:, 'ffnn_predicted_age'].values - shuffled_age.values)))
        total_permutations = pd.DataFrame.from_dict({'sq_err': total_permutation_mean_sq_err, 'perm_type': 'total'})
        #
        dph_permutations_mean_sq_err = []
        for permutation_index in range(n_permutations):
            by_dph = syll_data.groupby('dph')
            shuffled = by_dph.sample(frac=1)

            dph_permutations_mean_sq_err.append(
                np.mean(
                    np.square(
                        syll_data.loc[:, 'ffnn_predicted_age'].values - shuffled.loc[:, 'age'].values)))
        dph_permutations = pd.DataFrame.from_dict({'sq_err': dph_permutations_mean_sq_err, 'perm_type': 'dph'})
        permutations = pd.concat(objs=[no_permutation, total_permutations, dph_permutations], ignore_index=True)
        permutations.set_index('perm_type', append=True, inplace=True)
        perm_group_means = permutations.groupby(level='perm_type').mean()
        #
        perm_group_means = pd.concat({syll: perm_group_means}, names=['syll'])
        perm_group_means = pd.concat({animal: perm_group_means}, names=['bird'])
        permutations['bird'] = animal
        permutations['syll'] = syll
        if 'score' in locals():
            score = pd.concat(objs=[score, perm_group_means])
            all_permutations = pd.concat([all_permutations, permutations])
        else:
            score = perm_group_means
            all_permutations = permutations

score.to_pickle('tutored_agePrediction_MSE.pkl')
all_permutations.to_pickle('agePrediction_allPermutations_orig_architecture.pkl')
