function [syllTableOut,pcaInfo] = groupReduce(syllTableIn)

%{
Function takes a table of syllable renditions (syllTableIn) with at least
4 columns:
- age: a matlab duration variable for age at production time
- type: a matlab categorical variable indicating syllable type of
    rendition (A, B, etc)
- latent: a 32D vector of the rendition's likeliest location in vae latent
    space
- embed: 2D umap coordinates of syllable

Function returns a table of syllable renditions (syllTableOut) with
4 additional columns:
- segment_index: a number assigned to each row of the table is added if the
input table didn't have it
- pca: the syllable type-wise principal component coordinates of the
syllable in latent space
- pca_tsquared: Hotelling's t-squared statistic
- embed_normalized: z-scored UMAP coordinates, useful for plotting syllables
so they're distributed around (0,0)

Function also returns a second table (pcaInfo) with 1 row per principal
component, and 5 columns
- pca_coeff: the unit vector in original latent space pointing in PC
direction
- pca_explained: total variance explained by variance along this component
direction
- pca_var: variance along this component
- pc_number: rank order of this component (1 is the component with most
variation, 2 with second-most, etc)
- latent_means: the location of the syllable's mean location in latent
space. PCA coordinates are 0-centered, so this mean is required to
reconstruct the original latent location from PC coordinates

Function must be run in a folder containing a file called 'hatchdate.mat'
that contains a single variable expressing the bird's hatchdate as a matlab
datetime
%}

syllTableIn = unique(syllTableIn);
syllTableIn.Properties.UserData = struct('hatchdate',importdata('hatchdate.mat'));
if ~any(ismember('segment_index',syllTableIn.Properties.VariableNames))
    syllTableIn.segment_index(:) = 1:height(syllTableIn);
end

syllTableIn.type = removecats(syllTableIn.type);
pcaOut = rowfun(@myReduce,syllTableIn,...
    'InputVariables',{'latent','embed','age','segment_index'},...
    'GroupingVariables','type',...
    'NumOutputs',2,...
    'OutputVariableNames',{'pca_syllInfo','pca_statsInfo'});

pc_syllTable = vertcat(pcaOut.pca_syllInfo{:});

syllTableOut = join(syllTableIn,pc_syllTable,'keys',{'segment_index','age'});
pcaInfo = pcaOut(:,{'type','pca_statsInfo'});

end




function [syllInfo,statsInfo] = myReduce(latentsIn,embedIn,TimesIn,indexIn)

%centered PCA version
[pca_coeff,pca_score,pca_var,pca_tsquared,pca_explained,latent_means] = pca(latentsIn,'Centered',true);
statsInfo = table(pca_coeff',pca_explained,pca_var,(1:length(pca_explained))',latent_means',...
    'VariableNames',{'pca_coeff','pca_explained','pca_var','pc_number','latent_means'});
statsInfo = {statsInfo};
syllInfo = table(TimesIn,pca_score,pca_tsquared,indexIn,...
    'VariableNames',{'age','pca','pca_tsquared','segment_index'});

embedMeans = mean(embedIn);
embedCentered = embedIn - embedMeans;
embedDev = std(embedCentered(:));
embedNorm = embedCentered/embedDev;
syllInfo.embed_normalized(:,1:2) = embedNorm;

syllInfo = {syllInfo};
end