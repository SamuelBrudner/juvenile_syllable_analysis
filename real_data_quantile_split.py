import os
import pandas as pd
import numpy as np

"""
Combine predicted age data (written to text files preprocessing matlab routine) across animals.
Calculate predicted age quantiles and save to "quantile_savename"
Calculate quantile-level overnight shifts and save to "shifts_savename"
"""

parent_dir = 'predicted_age_tables' # this directory is created and filled in initial_preprocess.m
quantile_list = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
quantile_savename = f'real_data_pred_age_{len(quantile_list)}quantiles.pkl'
shifts_savename = f'real_data_pred_age_quantile_shifts_{len(quantile_list)}quantiles.pkl'

# combine all the predicted age information across animals
for animal in ['list', 'of', 'IDs']:
    print(animal)
    datafile = os.path.join(parent_dir, animal + '_predicted_age_table.txt')
    data_one_bird = pd.read_csv(datafile, sep='\t')
    if not 'data' in locals():
        data = data_one_bird
    else:
        data = pd.concat([data, data_one_bird])

# Filtering/feature calculation
data = data[data['age'] >= 60]
data = data[data['age'] < 95]
data = data[data['partition'] != 'Laser On']
data.loc[:, 'dph'] = np.floor(data.loc[:, 'age'])
data.loc[:, 'tod'] = data.loc[:, 'age'] - data.loc[:, 'dph']  # time of day
data.loc[:, 'tod_group'] = np.digitize(data.loc[:, 'tod'], np.linspace(0, 1, num=11))

#calculate quantile levels of performance
quantile_groups = data.groupby(by=['dph', 'tod_group', 'type', 'bird']).filter(
    lambda x: len(x) >= 30).groupby(
    by=['dph', 'tod_group', 'type', 'bird'])  # discard bins with fewer than 50 renditions
quantiles = quantile_groups['ffnn_predicted_age'].aggregate('quantile', q=quantile_list)
quantiles.index.names = ['dph', 'tod_group', 'type', 'bird', 'quantile']
quantiles = quantiles.to_frame()
quantile_age = quantile_groups['age'].aggregate('median')
quantiles = quantiles.join(quantile_age)
quantiles.sort_index(inplace=True)
quantiles.to_pickle(quantile_savename)


# Go through pairs of days to calculate quantile-level overnight shifts
plausible_inds = [([day, day + 1], syllable, bird)
                  for day in np.unique(quantiles.index.get_level_values('dph'))
                  for syllable in np.unique(quantiles.index.get_level_values('type'))
                  for bird in np.unique(quantiles.index.get_level_values('bird'))
                  ]
for pl_ind in plausible_inds:
    [day1, day2], syllable_type, bird = pl_ind
    pl_ind_reconstruction = ([day1, day2], slice(None), syllable_type, bird)
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
shifts = shifts.reorder_levels(['bird', 'type', 'pre_dph', 'quantile'])
shifts = shifts.loc[shifts.loc[:, 'age'] <= 0.6, :].reset_index()
shifts.to_pickle(shifts_savename)
shifts.to_csv(shifts_savename.replace('.pkl', '.txt'), sep='\t') # save to txt so we can read in matlab for linear mixed effects modeling
