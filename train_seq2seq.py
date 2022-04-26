import os
import json
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
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_subset: str = field(
        default=None,
        metadata={"help": ""}
    )
    context_column: str = field(
        default="text",
        metadata={"help": "The key in each dataset example containing the text"}
    )
    target_column: str = field(
        default="text",
        metadata={"help": "The key in each dataset example containing the text"}
    )
    context_max_length: int = field(
        default=256,
        metadata={"help": "Maximum number of tokens"}
    )
    target_max_length: int = field(
        default=256,
        metadata={"help": "Maximum number of tokens"}
    )
    train_limit_size: int = field(
        default=None,
        metadata={}
    )
    predict_key: str = field(
        default="test",
        metadata={"help": "The key in each dataset example containing the text"}
    )
    load_from_disk: bool = field(
        default=False
    )


def main():
    parser = transformers.HfArgumentParser((transformers.Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments))
        
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    
    transformers.set_seed(training_args.seed)

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.context_max_length
    )
    
    if data_args.load_from_disk:
        dataset = datasets.load_from_disk(data_args.dataset_name)
    else:
        dataset = datasets.load_dataset(data_args.dataset_name, data_args.dataset_subset)

    def tokenize_function(example):
        model_inputs = tokenizer(example[data_args.context_column], truncation=True, max_length=data_args.context_max_length)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(example[data_args.target_column], max_length=data_args.target_max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, padding=True)

    # Trainer
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if training_args.do_eval:
        metrics = trainer.evaluate(max_length=data_args.target_max_length)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predict_dataset = tokenized_dataset[data_args.predict_key]
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.json")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as f:
                json.dump(predictions, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
