import logging
import torch
import nltk
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm1

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed
from transformers.file_utils import is_offline_mode
from utils.arguments import parse_args
from multitask_model import MultitaskModel
from preprocess import convert_to_features
from multitask_data_collator import MultitaskTrainer, NLPDataCollator
from multitask_eval import multitask_eval_fn
from checkpoint_model import save_model
from pathlib import Path
import wandb
import os

logger = logging.getLogger(__name__)

wandb.init(mode="disabled")

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    args = parse_args()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    dataset_dict = {
        "BERTScore": load_dataset(
            "multitask_dataloader.py",
            data_files={
                "train": "../../partitions/BERTScore/BERTScore_train.tsv",
                "validation": "../../partitions/BERTScore/BERTScore_validate.tsv",
                },
            ),
        "CushLEPOR": load_dataset(
            "multitask_dataloader.py",
            data_files={
                "train": "../../partitions/CushLEPOR/CushLEPOR_train.tsv",
                "validation": "../../partitions/CushLEPOR/CushLEPOR_validate.tsv",
                },
            ),
        
        "COMET": load_dataset(
            "multitask_dataloader.py",
            data_files={
                "train": "../../partitions/COMET/COMET_norm_train.tsv",
                "validation": "../../partitions/COMET/COMET_norm_validate.tsv",
                },
            ),
        "TransQuest": load_dataset(
            "multitask_dataloader.py",
            data_files={
                "train": "../../partitions/TransQuest/TransQuest_norm_train.tsv",
                "validation": "../../partitions/TransQuest/TransQuest_norm_validate.tsv",
                },
            ),
        }

    for task_name, dataset in dataset_dict.items():
        print("Task name: ", task_name)
        print("Dataset: ", dataset_dict[task_name]["train"][0])
        print("\n\n\n")

    model_names = [args.model_name_or_path] * 4
    print(f"\n\n\nVar model_names: {model_names}\n\n\n")
    
    config_files = model_names
    for idx, task_name in enumerate(["BERTScore", "CushLEPOR", "COMET", "TransQuest"]): #Add all other keys to this dict
        model_file = Path(f"./{task_name}_model/pytorch_model.bin")
        config_file = Path(f"./{task_name}_model/config.json")
        if model_file.is_file():
            model_names[idx] = f"./{task_name}_model"

        if config_file.is_file():
            config_files[idx] = f"./{task_name}_model"

    multitask_model = MultitaskModel.create(
        model_name=model_names[0],
        model_type_dict={
            "BERTScore": transformers.AutoModelForSequenceClassification,
            "CushLEPOR": transformers.AutoModelForSequenceClassification,
            "COMET": transformers.AutoModelForSequenceClassification,
            "TransQuest": transformers.AutoModelForSequenceClassification,
            },
        model_config_dict={
            "BERTScore": transformers.AutoConfig.from_pretrained(model_names[0], num_labels=1),
            "CushLEPOR": transformers.AutoConfig.from_pretrained(model_names[0], num_labels=1),
            "COMET": transformers.AutoConfig.from_pretrained(model_names[0], num_labels=1),
            "TransQuest": transformers.AutoConfig.from_pretrained(model_names[0], num_labels=1),
            },
        )

    print("Encoder Word Embeddings: ", multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
    print("BERTScore Word Embeddings: ", multitask_model.taskmodels_dict["BERTScore"].roberta.embeddings.word_embeddings.weight.data_ptr())
    print("CushLEPOR Word Embeddings: ", multitask_model.taskmodels_dict["CushLEPOR"].roberta.embeddings.word_embeddings.weight.data_ptr())
    print("COMET Word Embeddings: ", multitask_model.taskmodels_dict["COMET"].roberta.embeddings.word_embeddings.weight.data_ptr())
    print("TransQuest Word Embeddings: ", multitask_model.taskmodels_dict["TransQuest"].roberta.embeddings.word_embeddings.weight.data_ptr())

    convert_func_dict = {
        "BERTScore": convert_to_features,
        "CushLEPOR": convert_to_features,
        
        "COMET": convert_to_features,
        "TransQuest": convert_to_features,
        
        }

    columns_dict = {
        "BERTScore": ["input_ids", "attention_mask", "labels"],
        "CushLEPOR": ["input_ids", "attention_mask", "labels"],
        "COMET": ["input_ids", "attention_mask", "labels"],
        "TransQuest": ["input_ids", "attention_mask", "labels"],
        }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )

    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in features_dict.items()
    }
    
    print(args.per_device_train_batch_size, type(args.per_device_train_batch_size))

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            learning_rate=5e-5,
            do_train=True,
            num_train_epochs=args.num_train_epochs,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=args.per_device_train_batch_size,
            report_to=None,
            save_steps=3000,
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
    )
    trainer.train()

    ## evaluate on given tasks
    multitask_eval_fn(multitask_model, "xlm-roberta-base", dataset_dict)

    ## save model for later use
    save_model("xlm-roberta-base", multitask_model)


if __name__ == "__main__":
    main()
