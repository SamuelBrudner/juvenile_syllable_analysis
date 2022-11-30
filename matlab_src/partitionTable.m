function myTable = partitionTable(myTable)
%{
Function takes a table of syllables (myTable) and returns the same table
with an extra column ("partition") taking the categorical values "Train" or
"Test." These values are used to partitions rows randomly into training and
test renditions for later analyses (predicted age, eg).

Because of its use in a variety of routines, function requires that input
contain a column called "laser" containing the categorical value "Off" --
rows with other "laser" values will be assigned neither Test nor Train
values, and can cause errors.
%}

TRAINING_FRACTION = 0.8;

myTable.partition(:) = categorical("Unassigned");

myTable.partition(myTable.laser==categorical("On")) = categorical("Laser On");
myTable.partition(myTable.laser==categorical("Partial")) = categorical("Laser Partial");
myTable.partition(myTable.laser==categorical("Off")) = categorical("Test"); % initialization
nOff = sum(myTable.laser==categorical("Off"));
nTrain = round(TRAINING_FRACTION * nOff);

trainInds = datasample(find(myTable.laser==categorical("Off")),nTrain,'Replace',false);
myTable.partition(trainInds) = categorical("Train");
