from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline
import os
from dotenv import load_dotenv
from datasets import load_dataset
from data_clean import *
import math
import argparse

load_dotenv()


def fine_tune(model_name, dataset, output_dir):
    if model_name == "AI-Sweden-Models/gpt-sw3-126m": 
        access_token = os.environ.get('ACCESS_TOKEN')
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token, mlm=False)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else: 
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    dataset = load_dataset('csv', data_files="/cluster/home/ingvlt/projects/master/master-project/genderSwapAP2019.csv")
    dataset = dataset['train'].train_test_split(test_size=0.2)
   
    # change column name if not 'text' 
    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]], padding=True, truncation=True)

    def tokenize_function(example):
        result = tokenizer(example['text'])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result


    # column name to be removed may not be named 'text', change if needed
    if model_name == "AI-Sweden-Models/gpt-sw3-126m": 
            tokenized_datasets = dataset.map(
            preprocess_function, 
            batched=True, 
            num_proc=4, 
            remove_columns=["text"]
            )
    else: 
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {}
        for k in examples.keys():
            concat = sum(examples[k], [])
            concatenated_examples[k] = concat

        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 3
    # Show the training loss with every epoch
    logging_steps = len(lm_datasets["train"]) // batch_size

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        # fp16=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=logging_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    eval_results = trainer.evaluate()
    print(f">>> Perplexity before: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.train()

    eval_results = trainer.evaluate()
    print(f">>> Perplexity after: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.save_model(output_dir)


def main(args): 
    fine_tune(args.model, args.data, args.file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m',
        '--model',
        dest='model',
        help='Model to fine-tune'
    )
    parser.add_argument(
        '-d',
        '--data',
        dest='data',
        help='Dataset to use for fine-tuning'
    )
    parser.add_argument(
        '-f', 
        '--file', 
        dest='file', 
        help='direcory/filname for saving model'
    )
    args = parser.parse_args()
    main(args)

