import os

import pandas as pd

"""
This file reads the predicted age of simulated data (predAgeFromSimulation.m) across syllables and birds,
merges that data into one table,
calculates quantile levels of performance,
calculates quantile-level overnight shifts in performance,
then saves those shifts as a tab-separated table (OUTPUT_FN)
"""
QUANTILE_LIST = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

DATA_BASENAME = 'simulated_pred_age_data.txt'

dataDict = {'animal': {"syllables": ['A', 'B'],  # sylls for this animal
                       "sim_data_parent": "path_to_choleskyModels_periodic"}}

OUTPUT_FN = 'simulation_overnight_shifts.txt'

######### collect simulated data predicted age across animals and syllables ###########
for animal, animal_dict in dataDict.items():
    print(animal)
    for model in ["orig"]:
        print(model)
        for syllable in animal_dict["syllables"]:
            print(syllable)
            read_dir = os.path.join(animal_dict["sim_data_parent"], syllable)
            pred_age_filename = os.path.join(read_dir, DATA_BASENAME)
            try:
                syllable_table = pd.read_csv(pred_age_filename, sep='\t')
            except FileNotFoundError:
                print('Could not read ' + pred_age_filename)
                syllable_table = pd.DataFrame()
            pred_age_filename_normEntropy = os.path.join(read_dir, '_'.join(['norm', DATA_BASENAME]))

            try:
                norm_syllable_table = pd.read_csv(pred_age_filename_normEntropy, sep='\t')
            except FileNotFoundError:
                print('Could not read ' + pred_age_filename_normEntropy)
                norm_syllable_table = pd.DataFrame()
            try:
                data_table = pd.concat([syllable_table, norm_syllable_table, data_table])
            except NameError:
                data_table = pd.concat([syllable_table, norm_syllable_table])
data_table = data_table[
    data_table.columns.drop(list(data_table.filter(regex='latent')))]  # We're not going to use latent info any more

######### bin data and calculate quantiles ###########
data_table = data_table[data_table['age'] >= 60]
data_table = data_table[data_table['age'] < 95]
data_table.loc[:, 'dph'] = np.floor(data_table.loc[:, 'age'])
data_table.loc[:, 'tod'] = data_table.loc[:, 'age'] - data_table.loc[:, 'dph']  # time of day
data_table.loc[:, 'tod_group'] = np.digitize(data_table.loc[:, 'tod'], np.linspace(0, 1, num=11))
quantile_groups = data_table.groupby(by=['dph', 'tod_group', 'type', 'bird', 'partition']).filter(
    lambda x: len(x) >= 50).groupby(
    by=['dph', 'tod_group', 'type', 'bird', 'partition'])
quantiles = quantile_groups['predicted_age'].aggregate('quantile', q=QUANTILE_LIST)
quantiles.index.names = ['dph', 'tod_group', 'type', 'bird', 'partition', 'quantile']
quantiles = quantiles.to_frame()
quantile_age = quantile_groups['age'].aggregate('median')
quantiles = quantiles.join(quantile_age)
quantiles.sort_index(inplace=True)

######### calculate overnight shifts at quantile level ###########
# each index retrieves consecutive days
plausible_inds = [([day, day + 1], syllable, bird, partition)
                  for day in np.unique(quantiles.index.get_level_values('dph'))
                  for syllable in np.unique(quantiles.index.get_level_values('type'))
                  for bird in np.unique(quantiles.index.get_level_values('bird'))
                  for partition in np.unique(quantiles.index.get_level_values('partition'))]
for pl_ind in plausible_inds:
    [day1, day2], syllable_type, bird, partition = pl_ind
    pl_ind_reconstruction = ([day1, day2], slice(None), syllable_type, bird, partition)
    try:
        subset_2day = quantiles.loc[pl_ind_reconstruction, :]
    except KeyError:
        continue
    if day1 < np.min(subset_2day.index.get_level_values(level='dph')):
        continue
    prior_evening_tod_group = np.max(
        subset_2day.loc[day1, :].index.get_level_values(level='tod_group'))
    if day2 > np.max(subset_2day.index.get_level_values(level='dph')):
        continue
    next_morning_tod_group = np.min(
        subset_2day.loc[day2].index.get_level_values(level='tod_group'))
    overnight_duration = next_morning_tod_group - prior_evening_tod_group
    if overnight_duration >= 6:
        continue

    prior_evening_pred_age = subset_2day.loc[([day1], prior_evening_tod_group, syllable_type, bird)].droplevel(
        level='tod_group')
    next_morning_pred_age = subset_2day.loc[([day2], next_morning_tod_group, syllable_type, bird)].droplevel(
        level='tod_group')
    data_subset = pd.concat([prior_evening_pred_age, next_morning_pred_age]).unstack(level='dph')
    data_subset.columns.set_names(['quantity', 'dph'], inplace=True)
    quantity_groups = data_subset.groupby(by='quantity', axis=1)
    shifts_temp = quantity_groups.apply(
        lambda x: x.loc[:, (slice(None), [day2])].droplevel('dph', axis=1) - x.loc[:, (slice(None), [day1])].droplevel(
            'dph', axis=1)).droplevel(level=0, axis=1)

    shifts_temp = pd.concat({day1: shifts_temp}, names=['pre_dph'])
    if 'shifts' in locals():
        shifts = pd.concat([shifts, shifts_temp])
    else:
        shifts = shifts_temp

shifts = shifts.reorder_levels(['partition', 'bird', 'type', 'pre_dph', 'quantile'])
shifts = shifts.loc[shifts.loc[:, 'age'] <= 0.6, :].reset_index()
shifts.to_csv(OUTPUT_FN, sep='\t')
