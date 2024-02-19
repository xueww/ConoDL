# ConoDL

ConoDL is a sequence-based deep learning framework designed for the rapid and large-scale generation and screening of artificial conotoxins, including the conotoxin generation model, ConoGen, and the conotoxin prediction model, ConoPred.



## Requirement
Bio==1.79
datasets==2.12.0
numpy==1.21.6
pandas==0.24.2
torch==1.12.1+cu102
tokenizers==0.10.3
transformers==4.15.0



## ConoGen
Change directories to the ConoGen folder `cd ConoGen`

### Download checkpoint
Download checkpoint from Zenodo, store it in the ConoGen_checkpoints folder and the ProGen_checkpoints folder, and change the checkpoint name to `pytorch_model.bin`

### training ConoGen
`python 1_fine-tuning.py`

### generation using ConoGen
1. generate the artificial conotoxin sequences with specified parameter values.
`python 2.1_sample.py --decoding_strategy 'top-p' --p 0.80 --t 0.70 --max-length 100 --num-samples 10 --repetition_penalty 1.5`
    `--decoding_strategy : the sampling strategy.`
    `--p : the probability threshold for top-p sampling.`
    `--t : the temperature parameter. It is used to control the variety of generated text.`
    `--max-length : the maximum length of sequence generated.`
    `--num-samples : the number of sequence generated.`
    `--repetition_penalty : the repetition penalty parameter. It is used to control the level of text duplication.`
    `--context : the context parameter. It is used to avoid meaningless text.`
2. batch generate artificial conotoxin sequences.
`python 2.2_sample.py --max-length 100 --num-samples 10000`



## ConoPred
Change directories to the ConoPred folder `cd ConoPred`

### training ConoPred
1. prepare dataset
    1.1 change directories to the datasets folder `cd dataset`
    1.2 prepare the conotoxin dataset for training wae model
    1.3 prepare positive and negative datasets for training classifier and divide the dataset into training and testing sets. `python create_classifier_dataset.py`
2. train the wae model. `python 1_train_wae.py`
3. training the classifier. `python 2_train_classifier.py`

### prediction using ConoPred
1. change directories to the prediction folder `cd prediction`, prepare the artificial conotoxin sequences.
2. predict artificial conotoxins and obtain probability scores. `python 3_predict.py`





