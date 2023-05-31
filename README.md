# Master-project :)

## Setup
1. Install the requirements: pip install -r requirements.txt
2. Create .env similar to .env-example

## Pronoun counting
The datasets are excluded from the code base due to the size.
- Norsk Aviskorpus (NAK) was downloaded 1st of Februrary 2023
- NorNE is downloaded using Huggingface
- Scandi-reddit is downloaded using Huggingface
- Universal Dependencies is downloaded using Huggingface

Counting the number of pronouns in Norsk Aviskorpus (NAK):
1. Download [NAK](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-4/)
2. Unzip all .tar.gz and .gz files
3. Run ```python scripts/make_dataset.py -c nak```

Counting the number of pronouns in reddit dataset: 
1. Run ```python scripts/make_dataset.py -c reddit```

Counting the number of pronouns in NorNE: 
1. Run ```python scripts/make_dataset.py -c norne```

The pronoun count is inspired by [Lossius and Ruud](https://github.com/andrinelo/norwegian-nlp/blob/main/experiments/pronoun_count/pronoun_count_norsk_aviskorpus.py).

## Gender-swapping and balancing dataset
Change input_data_dir in main.py to the path to your Norsk Aviskorpus folder.


Create a:
- gender-swapped NAK:
  1. Set the dictionary used in main.py to "PRONOUN_SWAP_DICT" 
  2. Run ```python scripts/make_dataset -n gender-swap```
- gender-neutral swapped NAK:
  1. Set the dictionary used in main.py to "GENDER_FLUID_PRONOUN_SWAP_DICT" 
  2. Run ```python scripts/make_dataset -n gender-swap```
- gender-swapped and anonymised NAK by running ```python scripts/make_dataset -n both```
- gender-swapped NorNE by running ```python scripts/make_dataset -o gender-swap```
- gender-swapped Scandi-Reddit by running ```python scripts/make_dataset -o gender-swap```

-r is for reddit, -n is for nak and -o is for norne, followed by the type of dataset you want.  
Options are: 
- gender-swap 
- anonymize 
- both 
- gender-balance 
- neutral-balance


## Fine-tuning models
To fine-tune a language model with one of the datasets:
1. Run ```python scripts/fine_tuning.py -m <model_name>  -d <dataset> -f <directory-name>```

## Performing POS
To perform POS:
1. Use the path of the model created in the step above
2. Set a name in main for output directory 
3. Run ```python scripts/pos.py -m <model_name> -f <output directory name> ```
Here, model-name should be the name of a directory containing a model


## Predictions 
Sentence to predict/generate from can be changed in predict.py  
Run predicitons with BERT-based model with masked language modelling:  
1. Run ```python scripts/predict.py -i <model_name> ``` For inference with a generative model OR
2. Run ```python scripts/predict.py -p <model_name> ``` For inference with a BERT-based model
model_name should be a directory containing the model you want to make predictions with


## Calculating bias
To calculate the bias of the model, the code from [Touileb et al.](https://github.com/SamiaTouileb/Biases-Norwegian-Multilingual-LMs) was used. 
1. Run the compute_scores.py file from [Touileb et al.](https://github.com/SamiaTouileb/Biases-Norwegian-Multilingual-LMs/blob/main/codes/compute_scores.py) with the models created in the experiments.
2. Add the files created after running compute_scores.py to a directory.
3. Update scripts/evaluation.py with the path to the directory.
4. Run scripts/evaluation.py to calculate the F1 macro score between the gold data from Statistics Norway and the directory. Gold data can be downloaded from the [GitHub](https://github.com/SamiaTouileb/Biases-Norwegian-Multilingual-LMs/tree/main/gold_data) of Touileb et al. Convert the file to a .csv-file.



