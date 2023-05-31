from data_clean import *
import os

PRONOUNS = ['han', 'ham', 'hun', 'ho', 'henne', 'hen']

pronoun_dict = {i: 0 for i in PRONOUNS}


def count_pronouns(string):
    pronoun_dict = {i: 0 for i in PRONOUNS}
    words = string.split(" ")
    count = 0
    other = 0
    for word in words:
        if word.lower() in PRONOUNS:
            pronoun_dict[word.lower()] += 1
            count += 1
        else:
            other += 1
    return count, other, pronoun_dict


def divide_genders(dict):
    male_pronouns = dict['han'] + dict['ham']
    female_pronouns = dict['ho'] + dict['hun'] + dict['henne']
    neutral_pronouns = dict['hen']
    return male_pronouns, female_pronouns, neutral_pronouns


def pronoun_counting_txt(file):
    text = read_file(file)
    pronouns, others = count_pronouns(text)
    total_pronouns = pronouns
    total_others = others
    total_words = total_pronouns + total_others
    male_pronouns, female_pronouns, neutral_pronouns = divide_genders(
        pronoun_dict)
    return total_pronouns, total_others, total_words, male_pronouns, female_pronouns, neutral_pronouns


def count_reddit():
    dataset = "alexandrainst/scandi-reddit"
    file = "orgReddit.txt"
    dataset_to_string(file, dataset)
    pronoun_counting_txt(file)


def count_norne():
    dataset = "NbAiLab/norne"
    file = "norneCount.txt"
    dataset_to_string(file, dataset)
    pronoun_counting_txt(file)


def count_nak(dir):
  total_pronouns = 0
  total_others = 0
  for root, dirs, files in os.walk(dir):
    for file in files:
      text = read_file(os.path.join(root, file))
      pronouns, others = count_pronouns(text)
      total_pronouns += pronouns
      total_others += others
  total_words = total_pronouns + total_others
  male_pronouns, female_pronouns = divide_genders(pronoun_dict)
  return total_pronouns, total_others, total_words, male_pronouns, female_pronouns


def get_gender_difference(file):
    pronouns, others, pronoun_dict = count_pronouns(file)
    if divide_genders(pronoun_dict) == (0, 0):
        return 0
    else:
        male_pronouns, female_pronouns = divide_genders(pronoun_dict)
        gender_diff = male_pronouns - female_pronouns
        return gender_diff


def get_neutral_gender_difference(file):
    pronouns, others, pronoun_dict = count_pronouns(file)
    if divide_genders(pronoun_dict) == (0, 0, 0):
        return 0, 0
    else:
        male_pronouns, female_pronouns, neutral_pronouns = divide_genders(
            pronoun_dict)
        gender_diff_m = male_pronouns - neutral_pronouns
        gender_diff_f = female_pronouns - neutral_pronouns
        return gender_diff_m, gender_diff_f
