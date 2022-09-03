This contains a train / evaluation set for Ultra-Fine entity typing project.

The portion of data derived from the Gigaword is excluded for copyright issues, so this is not exactly the dataset used in the paper but a subset of it. With this dataset and the model in the github repo (https://github.com/uwnlp/open_type), you should be able to reach similar, but slightly lower performance (~29F1 for new eval dataset, ~75MaF1 for ontonotes dataset). 

./crowd
contains crowdsourced examples. train_m.json is just train.json repeated multiple times.

./distant_supervision
contains distantly supervised training dataset, from entity linking (el_train.json, el_dev.json) and headwords (headword_train.json, headword_dev.json) 

./ontonotes
contains original ontonotes train/dev/test dataset from  https://github.com/shimaokasonse/NFGEC, as well as newly augmented training dataset. 

./ontology
contains other files, notably the type ontology containing 10331 types.
