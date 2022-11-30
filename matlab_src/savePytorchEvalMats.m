function dummyOut = savePytorchEvalMats(syllPcs,syllAge,syllType,birdId,pcaInfoTable)
%{
Function writes datasets needed to evaluate pytorch models that predict
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


eval_basename = fullfile('pytorch_eval',strcat(string(birdId), '_', string(syllType),'eval'));


%save n pcs
eval_writename = strcat(eval_basename, '_pcs.mat');
data=syllPcs(:,1:nPcs);
save(eval_writename,'data')


%save age
eval_writename = strcat(eval_basename, '_age.mat');
data = days(syllAge);
save(eval_writename,'data');

end