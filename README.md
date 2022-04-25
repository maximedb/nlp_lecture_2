Log-in on HuggingFace

```
huggingface-cli login
```

```
python train_fill_mask.py \
    --model_name_or_path distilbert-base-uncased \
    --output_dir pawsx_fill_mask \
    --dataset_name paws-x \
    --dataset_subset en \
    --do_train \
    --max_length 128 \
    --per_device_train_batch_size 128 \
    --text_column sentence1 \
    --num_train_epochs 10 \
    --report_to tensorboard \
    --logging_steps 25 \
    --warmup_ratio 0.1
```

```
python processing/concatenate_columns.py
```

```
python train_fill_mask.py \
    --model_name_or_path distilbert-base-uncased \
    --output_dir pawsx_fill_mask \
    --dataset_name data/text_dataset \
    --load_from_disk \
    --dataset_subset en \
    --do_train \
    --max_length 128 \
    --per_device_train_batch_size 128 \
    --text_column text \
    --num_train_epochs 10 \
    --report_to tensorboard \
    --logging_steps 25 \
    --warmup_ratio 0.1
```


```
python train_classifier.py \
    --model_name_or_path distilbert-base-uncased \
    --output_dir pawsx_classifier \
    --dataset_name paws-x \
    --dataset_subset en \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --max_length 128 \
    --per_device_train_batch_size 128 \
    --text_column_1 sentence1 \
    --text_column_2 sentence2 \
    --num_train_epochs 10 \
    --report_to tensorboard \
    --logging_steps 25 \
    --warmup_ratio 0.1 \
    --train_limit_size 1000
```

```
python train_fill_mask.py \
    --model_name_or_path bert-base-uncased \
    --output_dir pawsx_fill_mask \
    --dataset_name paws-x \
    --dataset_subset en \
    --do_train \
    --max_length 128 \
    --gradient_checkpointing \
    --per_device_train_batch_size 128 \
    --text_column sentence1 \
    --num_train_epochs 10 \
    --report_to tensorboard \
    --logging_steps 25 \
    --warmup_ratio 0.1 \
    --push_to_hub \
    --train_limit_size 1000
```


```
python train_seq2seq.py \
    --model_name_or_path facebook/bart-base \
    --output_dir outputs/pawsx-generative \
    --dataset_name data/generative_combined_dataset \
    --load_from_disk \
    --dataset_subset en \
    --do_train \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --context_max_length 128 \
    --target_max_length 128 \
    --gradient_checkpointing \
    --per_device_train_batch_size 64 \
    --context_column sentence1 \
    --target_column sentence2
```

Run the generative algorithm to generate new phrases.
```
python train_seq2seq.py \
    --model_name_or_path outputs/pawsx-generative \
    --output_dir outputs/pawsx-generative \
    --dataset_name data/generative_dataset \
    --load_from_disk \
    --dataset_subset en \
    --do_predict \
    --save_strategy epoch \
    --context_max_length 128 \
    --target_max_length 128 \
    --context_column sentence1 \
    --target_column sentence2 \
    --per_device_eval_batch_size 128 \
    --generation_num_beams 2 \
    --generation_max_length 128 \
    --predict_with_generate True \
    --predict_key train
```

- view metrics on HuggingFace
- compute_metrics
- use it within the pipeline
- upload a dataset to the hub
- upload a model to HugggingFace
