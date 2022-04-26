import os
import json
import numpy
import datasets
import transformers
from dataclasses import dataclass, field


os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        metadata={"help": "Path to dataset directory or dataset identifier from huggingface.co/datasets"}
    )
    dataset_subset: str = field(
        default=None,
        metadata={"help": "Optional dataset subset. For example, the glue dataset has many subsets."}
    )
    text_column_1: str = field(
        default="text",
        metadata={"help": "Column identifier for the column to perform classification (usually something like sentence/sentence1/text)"}
    )
    text_column_2: str = field(
        default=None,
        metadata={"help": "Column identifier for the second column to perform sentence pair classification (usually something like sentence2)"}
    )
    max_length: int = field(
        default=256,
        metadata={"help": "Maximum number of tokens in a sequence. The lower the less computationally intensive."}
    )
    predict_key: str = field(
        default="test",
        metadata={"help": "The key to use for prediction"}
    )
    load_from_disk: bool = field(
        default=False,
        metadata={"help": "Load a dataset saved to disk instead of from the hub"}
    )


def main():
    """ Main function to train a classifier on any given HuggingFace dataset """

    # Initiate a command-line argument parser with 3 sub-modules.
    # training_args -> see all possible options on https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    parser = transformers.HfArgumentParser((transformers.TrainingArguments, ModelArguments, DataTrainingArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    # set the seed for reproducability
    transformers.set_seed(training_args.seed)

    # load the dataset (and possible subset)
    if data_args.load_from_disk:
        dataset = datasets.load_from_disk(data_args.dataset_name)
    else:
        dataset = datasets.load_dataset(data_args.dataset_name, data_args.dataset_subset)
    # compute the number of different labels

    # load the model for sequence classification with the right number of labels
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path 
    )
    # load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        model_max_length=data_args.max_length
    )
    
    def tokenize_function(example):
        """ tokenize a single example or a batch of examples """
        # if there is no second column specifiec, do not use text_pair
        if data_args.text_column_2 is not None:
            # tokenize the text (and text_pair) with truncation to the max_length
            outputs = tokenizer(example[data_args.text_column_1], example[data_args.text_column_2], truncation=True, max_length=data_args.max_length)
        else:
            # tokenize the text with truncation to the max_length
            outputs = tokenizer(example[data_args.text_column_1], truncation=True, max_length=data_args.max_length)
        return outputs

    # tokenize the dataset to be readable by the model, batched=True is much faster, 1000 examples at a time.
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # initiate a collator to combine multiple sequences of different sizes with padding
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=True)

    # main object used to train the model, it encapsulte the training loops, placing tensors on devices, etc.
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # if needed, launch the training and save the model along with the results
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # if needed, evaluate the model on the eval_dataset specified in the trainer
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # if needed, perfrom some predictions and save the results somewhere
    if training_args.do_predict:
        predict_dataset = tokenized_dataset[data_args.predict_key]
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = numpy.argmax(predictions, axis=1).tolist()
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        
        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.json")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as f:
                json.dump(predictions, f, ensure_ascii=False)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path, 
        "tasks": "text-classification"
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
