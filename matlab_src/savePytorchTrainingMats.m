function dummyOut = savePytorchTrainingMats(syllPcs,syllAge,syllType,birdId,pcaInfoTable)
%{
Function writes datasets needed to train pytorch models that predict
distributions in latent space given age. It requires:
- syllPcs: a 32 pcs x M syllable renditions matrix expressing the PC coordinates of
syllables in latent space
- syllAge: a 1 x M syllable renditions vector of matlab durations, expressing the age
at production time of each syllable rendition
- syllType: a 1 x M syllable renditions categorical vector expressing the syllable
type (A, B, C, etc) of each rendition. THESE SHOULD ALL BE IDENTICAL
- birdId: a 1 x M syllable renditions categorical vector expressing the
bird id. THESE SHOULD ALL BE IDENTICAL.
- pcaInfoTable: this struct is the second output of groupReduce.m. To be
used here, it needs a "type" field matching this syllable type and it
needs a "n_pcs" field with a value indicating the smallest number of pcs
required to explain 99pct of this syllable's latent space variation.
%}

dummyOut = 1;

birdId = unique(birdId);
syllType = unique(syllType);

pcaInfo = pcaInfoTable(pcaInfoTable.type==syllType,:);
nPcs = pcaInfo.keep_nPcs;

train_basename = fullfile('pytorch_training',strcat(string(birdId), '_', string(syllType)));

%save age
writename = strcat(train_basename, '_age.mat');
data = days(syllAge);
save(writename,'data');

%save n pcs to cross 99% variance explained
writename = strcat(train_basename, '_pcs.mat');
data=syllPcs(:,1:nPcs);
save(writename,'data')

end