function [ffnn_pred_age, seg_in]= get_ffnn_predicted_age(latent_in,type_in,seg_in,ffnn_table_in)

%{
Function that returns predicted age values given query vae latent
locations, their corresponding syllable type, and a look-up table for
prediction networks by syllable type.
Inputs are:
- latent_in: 32D by N renditions matrix of vae latent locations
- type_in: a matlab categorical denoting the syllable type (A, B, C, etc)
- seg_in: a vector of numeric identifiers for the different query renditions
- ffnn_table_in: a table that contains rows for different syllable types
(listed in the "type" column") and corresponding networks (in the
"ffnn_l2a" column)
Output:
ffnn_pred_age: a vector of N predicted ages expressed as matlab durations
seg_in: unchanged from input variable seg_in, this value is returned to
facilitate matching the predicted age values to the associated renditions
outside this function
%}

type_ind = ffnn_table_in.type == type_in;

ffnn_l2a = ffnn_table_in.ffnn_l2a{type_ind};

ffnn_pred_age = days(ffnn_l2a(latent_in'));