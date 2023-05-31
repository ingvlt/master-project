import os
from data_clean import *
from gender_swap import *
from ner import *
import argparse
from predict import *
from pronoun_counting import *
from balance_dataset import balance_dataset, balance_gender_neutral_dataset, balance_reddit


def make_dataset(dataset_type: str):
    input_data_dir = "path_to_dir"
    for subdir, dirs, files in os.walk(input_data_dir):
        for file in files:
            string = read_file(os.path.join(subdir, file))
            clean_string = clean_html(os.path.join(subdir, file))
            if dataset_type == 'gender-swap':
                new_text = gender_swap(
                    clean_string, string, GENDER_FLUID_PRONOUN_SWAP_DICT)
                write_to_csv(new_text, 'neutral-swap.csv')
            elif dataset_type == 'anonymize':
                try:
                    new_text = anonymize(clean_string)
                    write_to_csv(new_text, "anonymized.csv")
                except:
                    continue
            elif dataset_type == 'both':
                # Remember to change to the dictionary you want to use
                # either GENDER_FLUID_PRONOUN_SWAP_DICT OR SWAP_DICT
                gender_swap_text = gender_swap(
                    clean_string, string, GENDER_FLUID_PRONOUN_SWAP_DICT)
                try:
                    new_text = anonymize(gender_swap_text)
                except:
                    continue
                write_to_csv(new_text, "gender_swap_anon.csv")
            elif dataset_type == 'gender-balance':
                gender_diff = get_gender_difference(clean_string)
                gender_balanced_text = balance_dataset(
                    gender_diff, clean_string)
                new_text = gender_balanced_text
                write_to_csv(new_text, 'balanced.csv')
            elif dataset_type == 'neutral-balance':
                gender_diff_m, gender_diff_f = get_neutral_gender_difference(
                    clean_string)
                gender_balanced_text = balance_gender_neutral_dataset(
                    gender_diff_m, gender_diff_f, clean_string)
                new_text = gender_balanced_text
                write_to_csv(new_text, 'neutral-balanced.csv')


def pronoun_count_helper(dataset_name):
    if dataset_name == "nak":
        path_to_nak = ""
        count_nak(path_to_nak)
    elif dataset_name == "reddit":
        count_reddit()
    elif dataset_name == "norne":
        count_norne()


def main(args):
    if args.nak:
        make_dataset(args.nak)
    elif args.norne: 
        gender_swap_norne()
    elif args.reddit: 
        if args.reddit == "gender-swap": 
            gender_swap_reddit()
        elif args.reddit == "gender-balance":
            balance_reddit("gender", "gender-balanced-reddit.csv")
        elif args.reddit == "neutral-balance":
            balance_reddit("neutral", "neutral-balanced-reddit.csv")
    elif args.count: 
        pronoun_count_helper(args.count)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-n',
        '--nak',
        dest='nak',
        help='Make dataset with NAK, either gender-swap, gender-neutral, gender-balance, neutral-balance, anonymize or both(swap and anonymize)'
    )
    parser.add_argument(
        '-r',
        '--reddit',
        dest='reddit',
        help='Make dataset with reddit, either gender-swap, gender-neutral, gender-balance, neutral-balance, anonymize or both(swap and anonymize)'
    )
    parser.add_argument(
        '-o',
        '--norne',
        dest='norne',
        action='store_true',
        help='Make dataset with norne, either gender-swap, gender-neutral, gender-balance, neutral-balance, anonymize or both(swap and anonymize)'
    )
    parser.add_argument(
        '-c', 
        '--count', 
        dest='count', 
        help='counts pronoun for given dataset'
    )

    args = parser.parse_args()
    main(args)
