import pandas as pd
from datasets import load_dataset


PRONOUN_SWAP_DICT = {'han': ['ho', 'hun'], 'ham': 'hun', 'hans': 'hennes', 'menn': 'kvinner',
                     'herr': 'fru', 'gutt': 'jente', 'gut': 'jente', 'guten': 'jenta',
                     'gutten': 'jenta', 'gutar': 'jenter', 'gutter': 'jenter', 'guttene': 'jentene',
                     'gutane': 'jentene', 'mann': 'kvinne', 'mennene': 'kvinnene', 'herrene': 'damene', 'mannen': ['kvinna', 'kvinnen'],
                     'herrer': 'damer', 'herrar': 'damer', 'ho': 'han', 'hun': ['han', 'ham'],
                     'hennes': 'hans', 'kvinna': 'mannen', 'kvinnen': 'mannen', 'kvinner': 'menn', 'fru': 'herr', 'jente': ['gut', 'gutt'],
                     'jenta': ['guten', 'gutten'], 'jenter': ['gutar', 'gutter'],
                     'jentene': ['gutane', 'guttene'], 'kvinne': 'mann', 'kvinnene': 'mennene',
                     'damene': 'herrene', 'damer': ['herrar', 'herrer']}

GENDER_FLUID_PRONOUN_SWAP_DICT = {'ho':'hen', 'hun':'hen', 'han':'hen', 'ham':'hen', 'henne':'hen'}


# gender swap NAK data
def gender_swap(clean_string, string, dictionary):
  words = string.split(" ")
  clean_words = clean_string.split(" ")
  swap_dict = dictionary
  if 'value="nno"/>' in words:
    for word in clean_words:
      if word.lower() in swap_dict:
        i = clean_words.index(word)
        if type(swap_dict[word.lower()]) is list:
          clean_words[i] = swap_dict[word.lower()][0]
        else:
          clean_words[i] = swap_dict[word.lower()]
  elif 'value="nob"/>' in words:
    for word in clean_words:
      if word.lower() in swap_dict:
        i = clean_words.index(word)
        if type(swap_dict[word.lower()]) is list:
          clean_words[i] = swap_dict[word.lower()][1]
        else:
          clean_words[i] = swap_dict[word.lower()]
  swapped_words = ' '.join(clean_words)
  return swapped_words


# gender swap data from NorNE
def gender_swap_norne():
    dataset_config = 'bokmaal'
    dataset = load_dataset("NbAiLab/norne", dataset_config)
    column_names = dataset['train'].column_names
    df = pd.DataFrame(data=dataset['train'], columns=column_names)
    # only tokens are used for pos
    tokenList = df['tokens'].values.tolist()
    for entry in tokenList:
        i = tokenList.index(entry)
        for word in entry:
            if word.lower() in PRONOUN_SWAP_DICT:
                j = entry.index(word)
                if type(PRONOUN_SWAP_DICT[word.lower()]) is list:
                    if dataset_config == 'bokmaal':
                        tokenList[i][j] = PRONOUN_SWAP_DICT[word.lower()][1]
                    else:
                        tokenList[i][j] = PRONOUN_SWAP_DICT[word.lower()][0]
                else:
                    tokenList[i][j] = PRONOUN_SWAP_DICT[word.lower()]

    df['tokens'] = tokenList
    df.to_csv("norne-swap.csv")


# gender swap data from reddit
def gender_swap_reddit(): 
    dataset = load_dataset("alexandrainst/scandi-reddit", split="train[:5000]")
    column_names = dataset.column_names
    df = pd.DataFrame(data=dataset, columns=column_names)
    docList = df["doc"].values.tolist()
    for entry in docList:
        i = docList.index(entry)
        wordList = entry.split(" ")
        for word in wordList:
            
            j = wordList.index(word)
            if word.lower() in PRONOUN_SWAP_DICT:
                if type(PRONOUN_SWAP_DICT[word.lower()]) is list:
                    wordList[j]= PRONOUN_SWAP_DICT[word.lower()][1]
                    
                else:
                    wordList[j] = PRONOUN_SWAP_DICT[word.lower()]
            docList[i] = " ".join(wordList)
    df['doc'] = docList
    df.to_csv("reddit-swap.csv")
