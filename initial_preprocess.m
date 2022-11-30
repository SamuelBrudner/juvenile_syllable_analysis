%{
Script to process vae datasets for 1 bird. To run, user needs to have saved
a matlab table (below, this is "datatable") with a row for every syllable
and the following columns:
- age: matlab duration variable indicating the age at syllable production
time
- type: categorical expressing syllable type (A, B, C, etc); the value
"Unassigned" will be skipped by subsequent analysis
- datetime: matlab datetime variable indicating syllable production time
- file: a .wav name of original song recording file
- dph: dph at syllable production time (floor of age)
- duration: matlab duration recording the syllable duration
- latent: a 32D vector of vae latents; the likeliest latent space location
for the syllable
- embed: UMAP coordinates based on the syllable's latent space location
- bird: matlab categorical with the bird's ID

This script generates several bird-specific files that are used in
downstream analysis. Use a file naming system so files for bird 1 are not
overwritten when running this step on bird 2.

%}

% SUBSCRIPT FOLDER must contain 'groupReduce.m' 'partitionTable.m'
% 'nnRegressSyllable2.m' 'get_ffnn_predicted_age.m'
SUBSCRIPT_FOLDER = fullfile('path','to','analysis','subscripts'); %eg matlab_src

% Where should data be saved?
OUTPUT_DIRECTORY = fullfile('path','to','output','directory');

DATA_TABLE_PATH = fullfile('path','to','datatable.mat'); % described in header

HATCHDATE_PATH = fullfile('path','to','hatchdate.mat');% single var expressing hatchdate as matlab datetime

addpath(SUBSCRIPT_FOLDER)

% Import datatable and calculate syllable-specific PCA
data_table = importdata(DATA_TABLE_PATH);
load(HATCHDATE_PATH);
[data_table, data_pcaInfo] = groupReduce(data_table);
data_pcaInfo.keep_nPcs =...
    rowfun(@(x) find(cumsum(x.pca_explained)>99,1),...
    data_pcaInfo,'InputVariables',...
    'pca_statsInfo',...
    'ExtractCellContents',true,...
    'OutputFormat','uniform');

pca_info_path = fullfile(OUTPUT_DIRECTORY, 'pcaInfo.mat');
save(pca_info_path,'data_pcaInfo');

% Laser inds are expected downstream so just toss 10 random rows
% Then assign training and test partitions randomly
laser_inds = randperm(height(data_table),10);
data_table.laser(:) = categorical("Off");
data_table.laser(laser_inds) = categorical("On");
data_table = partitionTable(data_table);

% Train neural networks to predict age from latent locations
% Save trained networks in the ffnn table
holdout_range = timerange(seconds(0),seconds(0));
ffnnFunc = @(age_in,seg_in,lat_in) nnRegressSyllable2(age_in,seg_in,lat_in,holdout_range);
trainset = data_table.partition == categorical("Train");
ffnn_table = rowfun(ffnnFunc,data_table(trainset,:),'InputVariables',{'age','segment_index','latent'},...
    'GroupingVariables', 'type',...
    'OutputFormat','table',...
    'NumOutputs',4,...
    'OutputVariableNames',{'ffnn_a2l','ffnn_info_a2l','ffnn_l2a','ffnn_info_l2a'});

ffnn_table_path = fullfile(OUTPUT_DIRECTORY,'ffnn_table.mat');
save(ffnn_table_path,'ffnn_table')

% Use trained neural networks to get predicted age for syllables in table
% Save table with this additional info at
% OUTPUT_DIRECTORY/processed_datatable.mat
get_ffnn_pred_age = @(l_in,t_in,s_in) get_ffnn_predicted_age(l_in,t_in,s_in,ffnn_table);
ffnn_pred_ages = rowfun(get_ffnn_pred_age,data_table,...
    'InputVariables',{'latent','type','segment_index'},...
    'OutputFormat','table',...
    'NumOutputs',2,...
    'OutputVariableNames',{'ffnn_predicted_age','segment_index'});
data_table = innerjoin(data_table,ffnn_pred_ages);

write_name = fullfile(OUTPUT_DIRECTORY,'processed_datatable.mat');
save(write_name,'data_table');

% Performance assessed in python, so save a txt file:
pred_age_table = data_table(:,['age','partition','type','bird','ffnn_predicted_age']);
if any(isduration(pred_age_table.age))
    pred_age_table.age = days(pred_age_table.age);
end
if any(isduration(pred_age_table.ffnn_predicted_age))
    pred_age_table.ffnn_predicted_age= days(pred_age_table.ffnn_predicted_age);
end
write_fn = string(pred_age_table.bird(0)+"_predicted_age_table.txt");
write_location = fullfile('predicted_age_tables',write_fn);
writetable(pred_age_table,write_location,"Delimiter","\t")


% Save training and test datasets in preparation for use by pytorch models
% of latent space distribution over time
trainset = data_table.partition == categorical("Train");
training_directory = fullfile(OUTPUT_DIRECTORY,'pytorch_training');
if ~exist(training_directory,'dir')
    mkdir(training_directory);
end
data_pcaInfo = importdata(pca_info_path);
dummyOutput =...
    rowfun(@(em,pc,age,type,bird) savePytorchTrainingMats(em,pc,age,type,bird,data_pcaInfo),...
    data_table(trainset,:),...
    'InputVariables',{'embed_normalized','pca','age','type','bird'},...
    'GroupingVariables','type',...
    'OutputVariableNames',{'bash_table'});

testset = data_table.partition == categorical("Test");
test_directory = fullfile(OUTPUT_DIRECTORY,'pytorch_eval');
if ~exist(test_directory,'dir')
    mkdir(test_directory);
end

dummyOutput = rowfun(@(pc,age,type,bird)saveBnafEvalMats(pc,age,type,bird,data_pcaInfo),...
    data_table(testset,:),...
    'InputVariables',{'pca','age','type','bird'},...
    'GroupingVariables','type',...
    'OutputVariableNames',{'bash_table'});

% Next, train gaussian models in pytorch
