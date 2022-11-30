function [syllNet_age2latent,tr_a2l,syllNet_latent2age,tr_l2a] = nnRegressSyllable2(ageIn,segIn,latentIn,holdout_ranges)

%{
Function trains neural networks to predict age at syllable rendition from
acoustic features of syllable expressed as a vector in vae latent space. It
also trains networks predicting location in latent space as a function of
age.
Input variables are:
- ageIn: a vector of N syllable rendition ages associated with renditions of
syllable type
- segIn: a numeric identifier unique to each rendition
- latentIn: the 32 latent features by N syllable renditions
- holdout_ranges: a cell array of matlab timerange variables. Syllables
produced at ages falling in these timeranges are not used during neural
network training.

Outputs are:
- syllNet_age2latent: a neural network that returns average latent space
locations given age inputs
- tr_a2l: a structure of information about the training process for
syllNet_age2latent networks
- syllNet_latent2age: a neural network that returns a predicted age given a
location in vae latent space
- tr_l2a: a structure of information about the training process for
syllNet_latent2age networks
%}

if ~isduration(ageIn(1))
    ageIn=days(ageIn);
end
syllableTable = timetable(ageIn,segIn,latentIn,'VariableNames',{'segment_index','latent'});
syllableTable.Properties.DimensionNames = {'age','Variables'};
if iscell(holdout_ranges)
    heldout_segments = cellfun(@(holdout_range) syllableTable{holdout_range,'segment_index'},holdout_ranges,'UniformOutput',false);
    heldout_segments = vertcat(heldout_segments{:});
    heldout_segments = unique(heldout_segments);
else
    heldout_segments = syllableTable{holdout_ranges,'segment_index'};
end
heldout_inds = ismember(syllableTable.segment_index,heldout_segments);
train_inds = ~heldout_inds;
syllableTable = timetable2table(syllableTable(train_inds,:));


syllNet_age2latent = fitnet([4 8 16],'trainbr');
syllNet_latent2age = fitnet([16 8 4],'trainbr');

syllNet_latent2age.trainParam.time = 1800; % Do not train more than 30 min
syllNet_latent2age.trainParam.max_fail = 3;
syllNet_age2latent.trainParam.max_fail = 3;


[syllNet_age2latent,tr_a2l] = train(syllNet_age2latent,days(syllableTable.age'),syllableTable.latent');
[syllNet_latent2age,tr_l2a] = train(syllNet_latent2age,syllableTable.latent',days(syllableTable.age'));
syllNet_latent2age = {syllNet_latent2age};


end