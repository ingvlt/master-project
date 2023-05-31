import requests
from dotenv import load_dotenv
from data_clean import *
from gender_swap import *
import os

load_dotenv()


# API-endpoint for named entity recognition from NB AI-lab
def ner_apicall(text: str):
    url = os.environ.get('URL')
    body = {
        "text": text,
        "group_entities": "false",
        "wait": "true"
    }
    x = requests.post(url, json=body)
    return x.json()


# function to anonymize named person entities
def anonymize(clean_string: str):
    ner_json = ner_apicall(clean_string)
    per_enitities = []

    # find all person entities from NER-call and append to new list of persons
    for l in ner_json['result']:
        if l['entity_group'] == 'PER':
            per_enitities = [x.lower() for x in per_enitities]
            if l['word'].lower() not in per_enitities:
                per_enitities.append(l['word'])

    # make new dict to store name of persons and their anonymous nickname
    new_text = clean_string.lower()
    person_dict = {}
    counter = 1
    for person in per_enitities:
        if person not in person_dict:
            person_dict.update({person: f'p{counter}'})
            counter += 1

    # change name of person entities to anonymized name from dict
    for k, v in person_dict.items():
        new_text = new_text.replace(k, v)

    return new_text

