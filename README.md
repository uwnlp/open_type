This repository contains code for the following paper:

## Ultra-Fine Entity Typing

#### Eusol Choi, Omer Levy, Yejin Choi and Luke Zettlemoyer. (ACL 2018)

Project website: https://homes.cs.washington.edu/~eunsol/_site/open_entity.html

### Dependencies:
- Pytorch (ver 0.3.0)
- Python3 
- Numpy
- Tensorboard 
- Pretrained word embeddings:
    Download "Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)" from https://nlp.stanford.edu/projects/glove/
    or do "wget http://nlp.stanford.edu/data/glove.840B.300d.zip".

### Configuration:
- You have to put set three paths at
  ./resources/constant.py
  
  FILE_ROOT=where you our dataset.
  
  GLOVE_VEC=the path where you can find pretrained glove vectors.
  
  EXP_ROOT=where you save models.

### Preprocessing:

 - The model reported in the paper is trained on a data from
    (1) a subset of Gigaword corpus, (2) Wikilink dataset, (3) Wikipedia document and (4) Indomain crowd-sourced data

  (2), (3), (4) can be downloaded from here http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz 
    
 - Gigaword is a licensed dataset from LDC, so is not released with the code. 
 - Without it, however, model can reach reasonable performances (29.8F1 instead of 31.7F1 reported).

 - Alternatively, you can email the first author get the processed version after verifying your LDC license.

### To train a model:

python3 main.py MODEL_ID -lstm_type single -enhanced_mention -data_setup joint -add_crowd -multitask

To train model on the Ontonotes dataset
python3 main.py onto -lstm_type single -goal onto  -enhanced_mention

To run predictions of pre-trained model:
python3 main.py MODEL_ID -lstm_type single -enhanced_mention -data_setup joint -add_crowd -multitask -mode test -reload_model_name MODEL_NAME_TIMESTAMP -eval_data crowd/test.json -load

### Scorer: 

python3 scrorer.py OUTPUT_FILENAME



Contact:
   Eunsol Choi -- firstname@cs.washington.edu
   
 
Credit:
- Some code is modified from existing code resources.
  * https://github.com/shimaokasonse/NFGEC
  * https://github.com/allenai/allennlp
