from transformers import pipeline
from data_clean import write_to_file
import argparse

# For BERT-based models
def predict_masked_word(model):
    nlp = pipeline('fill-mask', model=model)
    # change sentence to what you want
    sentence = f'Hen jobber som {nlp.tokenizer.mask_token} '
    write_to_file(str(nlp(sentence)), 'predict.txt')
    return nlp(sentence)


# For GPT-based models
def inference_causal_lm(model): 
    # change prompt to what you want
    prompt = "Hen jobber som "
    generator = pipeline("text-generation", model=model)
    text = generator(prompt)
    write_to_file(text, 'inference.txt')

def main(args): 
    if args.inference: 
        inference_causal_lm(args.inference)
    elif args.predict: 
        predict_masked_word(args.predict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i',
        '--inference',
        dest='inference',
        help='inference with the given model'
    )
    parser.add_argument(
        '-p', 
        '--predict', 
        dest='predict', 
        help='predict masked word with given model'
    )
    args = parser.parse_args()
    main(args)