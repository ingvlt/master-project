from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, Trainer, TrainingArguments, AutoTokenizer
import evaluate
import numpy as np 
import argparse

# Part-of-speech tagging
def pos(model_name, output_dir):
    # name of fine-tuned model 
    # use GPT-neo tokenizer with GPT-SW3, as GPT-SW3 does not have a fast tokenizer
    if model_name == "gpt-sw3":
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m', add_prefix_space=True, mlm=False)
        tokenizer.pad_token = tokenizer.eos_token
    else: 
       tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # using universal dependencies for all models
    dataset = load_dataset('universal_dependencies', 'no_bokmaal')
    label_list = list(set([tag for example in dataset["train"] for tag in example["upos"]]))
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    max_length = 256
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            # is_split_into_words=True,
            return_overflowing_tokens=True
        )
        labels = []
        for i, label in enumerate(examples['upos']):
            word_ids = tokenized_inputs.word_ids(batch_index=0)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
              if word_idx is None:
                  label_ids.append(-100)
              elif word_idx != previous_word_idx:
                if len(label)-1 < word_idx:
                  label_ids.append(-100)
                else:
                  label_ids.append(label_list.index(label[word_idx]))
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=4,
        remove_columns=['text', 'tokens', 'lemmas', 'head', 'deprel', 'deps', 'misc', 'upos', 'xpos', 'feats'],
    )
    

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
    
    metric = evaluate.load("seqeval")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
    
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    
        true_pred = []
        for el in true_predictions:
          l = list(map(str, el))
          true_pred.append(l)
    
        true_lab = []
        for el in true_labels:
          l = list(map(str, el))
          true_lab.append(l)
    
        results = metric.compute(predictions=true_pred, references=true_lab)
    
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy = "epoch",
        #eval_steps = 100,
       # save_steps = 500,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        # per_device_train_batch_size=16,
        # per_device_eval_batch_size=64,
        num_train_epochs=20,
        weight_decay=0.01,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()

def main(args): 
    pos(args.model, args.file)

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
        '-f', 
        '--file', 
        dest='file', 
        help='direcory/filname for saving model'
    )
    args = parser.parse_args()
    main(args)
    