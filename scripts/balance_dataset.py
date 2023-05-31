from datasets import load_dataset
import pandas as pd 
from pronoun_counting import get_neutral_gender_difference, get_gender_difference

MALE_PRONOUNS = ['han', 'ham']
FEMALE_PRONOUNS = ['ho', 'henne', 'hun']
NEUTRAL_PRONOUNS = ['hen']

# for balacing a dataset with the same amount of male and female pronouns
def balance_dataset(gender_diff, string):
  words = string.split(" ")
  count = 0
  if gender_diff > 0:
    for word in words:
        if count <= gender_diff:
            if word.lower() in MALE_PRONOUNS:
                count += 1
                i = words.index(word)
                words[i] = ''  
    balanced_words = ' '.join(words)
  else:
    for word in words:
      if count <= gender_diff:
        if word.lower() in FEMALE_PRONOUNS:
          count += 1
          i = words.index(word)
          words[i] = ''  
    balanced_words = ' '.join(words)
  return balanced_words

# balancing a dataset with the the same amount of gender neutral "hen" 
# and male and female pronouns
def balance_gender_neutral_dataset(gender_diff_m, gender_diff_f, string):
  words = string.split(" ")
  count = 0
  if gender_diff_f > 0:
    for word in words:
        if count <= gender_diff_f:
            if word.lower() in FEMALE_PRONOUNS:
                count += 1
                i = words.index(word)
                words[i] = ''  
    balanced_words = ' '.join(words)
  elif gender_diff_m > 0:
    for word in words:
      if count <= gender_diff_m:
        if word.lower() in MALE_PRONOUNS:
          count += 1
          i = words.index(word)
          words[i] = ''  
    balanced_words = ' '.join(words)
  else:
    for word in words:
      if count <= gender_diff_m:
        if word.lower() in NEUTRAL_PRONOUNS:
          count += 1
          i = words.index(word)
          words[i] = ''  
    balanced_words = ' '.join(words)    
  return balanced_words

# balance reddit dataset by using pandas DataFrame
def balance_reddit(type: str, file_name: str):
  dataset = load_dataset("alexandrainst/scandi-reddit", split="train[:5000]")
  column_names = dataset.column_names
  df = pd.DataFrame(data=dataset, columns=column_names)
  docList = df["doc"].values.tolist()
  for entry in docList:
    if type == "neutral":
      gender_diff_m, gender_diff_f = get_neutral_gender_difference(entry)
      balanced_text = balance_gender_neutral_dataset(gender_diff_m, gender_diff_f, entry)
    else: 
      gender_diff = get_gender_difference(entry)
      balanced_text = balance_dataset(gender_diff, entry)
  df['doc'] = balanced_text
  df.to_csv(file_name)



