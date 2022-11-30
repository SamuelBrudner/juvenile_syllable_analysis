%{
This script goes through birds and syllables types, loading for each type
corresponding "syllables" simulated in baseline or fixed entropy models
(see write_simulated_syllables.py). It uses these simulations and the
predicted age networks trained in initial_process.m to score the maturity
of these simulated renditions.
%}

dataDict = containers.Map(["animal_id"], ...
    {containers.Map(["syllables","pcaInfoPath","ffnnPath","cholesky_directory"], ...
    {["A","B", "C"], ... % list of syllable names occuring for the animal
    "path_to_pcaInfo.mat", ... % saved during 'initial_preprocess.m'
    "path_to_ffnn_table.mat", ... % OUTPUT_DIRECTORY/ffnn_table.mat in 'initial_preprocess.m'
    "path_to_choleskyNet_periodic" ... % houses simulation data, see write_simulated_syllables.py
    })});

orig_state = warning;
for animal = keys(dataDict)
    animal = animal{1};
    disp(animal)
    animalDict = dataDict(animal);
    syllList = animalDict("syllables");
    pcaInfo_fn = animalDict("pcaInfoPath");
    pcaInfo = importdata(pcaInfo_fn);
    predAgeNet_fn = animalDict("ffnnPath");
    warning('off','all')
    predAgeNets = importdata(predAgeNet_fn);
    warning(orig_state)
    for syll = syllList
        disp(syll)
        sim_superdir = animalDict("cholesky_directory");
        sim_basename = 'simulated_data.mat'; % set SIMULATION_FN in write_simulated_syllables.py
        sim_fn = fullfile(sim_superdir,...
            syll,sim_basename);
        sim_data = load(sim_fn);
        sim_pca = sim_data.pc_draw;
        nPcs = size(sim_pca,2);
        nObs = size(sim_pca,1);
        syllPca = pcaInfo.pca_statsInfo{pcaInfo.type==syll};
        pca_coeff = syllPca.pca_coeff;
        sim_lat_centered = sim_pca * pca_coeff(1:nPcs,:);
        sim_lat = sim_lat_centered + repmat(syllPca.latent_means',nObs,1);
        sim_table = table((1:nObs)',sim_lat, ...
            'VariableNames',{'segment_index','latent'});
        
        get_pred_age = @(lat,seg) get_ffnn_predicted_age(lat,syll,seg,predAgeNets);
        pred_age_table = rowfun(get_pred_age,sim_table, ...
            'InputVariables',{'latent','segment_index'}, ...
            'OutputFormat','table', ...
            'OutputVariableNames',{'predicted_age','segment_index'});
        sim_table = innerjoin(sim_table,pred_age_table, ...
            "Keys",'segment_index', ...
            'RightVariables','predicted_age');
        sim_table.predicted_age = days(sim_table.predicted_age);
        sim_table.age = sim_data.age_dph;
        save_dir = fullfile(sim_superdir,...
            syll);
        save_loc = fullfile(save_dir,'simulated_pred_age_data.mat');
        save(save_loc,"sim_table")
        save_loc = strrep(save_loc,'.mat','.txt');
        sim_table.partition(:)=categorical("simulation_free");
        sim_table.type(:)=categorical(string(syll));
        sim_table.bird(:)=categorical(string(animal));
        writetable(sim_table,save_loc,'Delimiter','tab','FileType','text');

        % Do it again for "fixed entropy"
        sim_basename = replace(sim_basename,".mat","_fixedEntropy.mat");

        sim_fn = fullfile(sim_superdir,...
            syll,sim_basename);
        sim_data = load(sim_fn);
        sim_pca = sim_data.pc_draw;
        nPcs = size(sim_pca,2);
        nObs = size(sim_pca,1);
        syllPca = pcaInfo.pca_statsInfo{pcaInfo.type==syll};
        pca_coeff = syllPca.pca_coeff;
        sim_lat_centered = sim_pca * pca_coeff(1:nPcs,:);
        sim_lat = sim_lat_centered + repmat(syllPca.latent_means',nObs,1);
        sim_table = table((1:nObs)',sim_lat, ...
            'VariableNames',{'segment_index','latent'});
        
        get_pred_age = @(lat,seg) get_ffnn_predicted_age(lat,syll,seg,predAgeNets);
        pred_age_table = rowfun(get_pred_age,sim_table, ...
            'InputVariables',{'latent','segment_index'}, ...
            'OutputFormat','table', ...
            'OutputVariableNames',{'predicted_age','segment_index'});
        sim_table = innerjoin(sim_table,pred_age_table, ...
            "Keys",'segment_index', ...
            'RightVariables','predicted_age');
        sim_table.predicted_age = days(sim_table.predicted_age);
        sim_table.age = sim_data.age_dph;
        save_loc = fullfile(save_dir,'norm_simulated_pred_age_data.mat');
        save_loc = strrep(save_loc,'.mat','.txt');
        sim_table.partition(:)=categorical("simulation_fixedEntropy");
        sim_table.type(:)=categorical(string(syll));
        sim_table.bird(:)=categorical(string(animal));
        writetable(sim_table,save_loc,'Delimiter','tab','FileType','text');
    end
end
