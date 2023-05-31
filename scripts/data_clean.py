import re
import codecs
import pandas as pd 
import numpy as np
import csv
from datasets import load_dataset

CLEANR = re.compile(r'<.*?>|\n|\r') 
# ISO-8859-1 - latin 1
# reading file, preserves all tags and metainformation
def read_file(file): 
 with codecs.open(file, 'r', encoding='utf-8') as f:
        print('Reading file...')
        lines = f.readlines()
        new_list = []
        for line in lines: 
          new_list.append(line.replace("\n"," ").replace('\r', ' '))
        stripped_text = ''.join(new_list)
        return stripped_text
 
# from csv to txt-file 
# for pronoun counting
def csv_to_txt(file):
  csv_file = file
  txt_file = "/cluster/home/ingvlt/projects/master/master-project/orgReddit.txt"
  with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r", encoding="utf-8") as my_input_file:
        print(my_input_file)
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

# clean file, removes html tags and makes clean text
def clean_html(file):
  with codecs.open(file, 'r', encoding='utf-8') as f:
    lines = f.read() 
    cleantext= re.sub(CLEANR, '', lines)
  return cleantext

def write_to_file(text, name):
  with codecs.open(name, 'a', 'utf-8') as f:
    print('Writing file...')
    f.write(text)

def write_to_csv(text, name):
  data = {
    'text': [text]
  }
  df = pd.DataFrame(data)
  # remove None values
  df['text'].replace('', np.nan, inplace=True)
  df.dropna()
  df.to_csv(name, mode='a', index=False)

# from huggingface dataset type to string
# used for pronoun counting
def dataset_to_string(file, dataset_name): 
  dataset = load_dataset(dataset_name, "bokmaal")
  column_names = dataset["train"].column_names
  df = pd.DataFrame(data=dataset["train"], columns=column_names)
  df.to_string(file)
