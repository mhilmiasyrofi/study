## Source
## - https://huggingface.co/microsoft/codebert-base
## - https://github.com/microsoft/CodeBERT
## - https://rsilveira79.github.io/fermenting_gradients/machine_learning/nlp/pytorch/text_classification_roberta/
## - https://www.thepythoncode.com/article/finetuning-bert-using-huggingface-transformers-python

import numpy as np
import random
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers.optimization import Adafactor, AdafactorSchedule



def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available


def read_actionable_warning_dataset():
    target_names = ["open", "close"]
    df_train = pd.read_csv("../data-derived/train.csv")
    df_test = pd.read_csv("../data-derived/test.csv")

    df_train["labels"] = df_train["category"].apply(lambda x: 0 if x == "open" else 1)
    df_test["labels"] = df_test["category"].apply(lambda x: 0 if x == "open" else 1)

    return (list(df_train["method_content"]), list(df_test["method_content"]), list(df_train["labels"]),  list(df_test["labels"])), target_names
    


class ActionableWarningDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate metric using sklearn's function
    acc = sklearn.metrics.accuracy_score(labels, preds)
    precision = sklearn.metrics.precision_score(labels, preds)
    recall = sklearn.metrics.recall_score(labels, preds)
    f1 = sklearn.metrics.f1_score(labels, preds)
    return {
        'accuracy': acc, 
        'precision': precision,
        'recall': recall,
        'f1' :f1
    }


if __name__ == "__main__":
    set_seed(1)

    # the model we gonna train, microsoft/codebert-base
    model_name = "microsoft/codebert-base"

    # max sequence length for each snippet code sample
    max_length = 512

    # call the function
    (train_texts, valid_texts, train_labels, valid_labels), target_names = read_actionable_warning_dataset()

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name)

    # tokenize the dataset, truncate when passed `max_length`,
    # and pad with 0's when less than `max_length`
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(
        valid_texts, truncation=True, padding=True, max_length=max_length)

    # convert our tokenized data into a torch Dataset
    train_dataset = ActionableWarningDataset(train_encodings, train_labels)
    valid_dataset = ActionableWarningDataset(valid_encodings, valid_labels)

    # load the model and pass to CUDA
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=len(target_names)).to("cuda")

    training_args = TrainingArguments(
        output_dir='./../codebert-checkpoint',          # output directory
        num_train_epochs=20,              # total number of training epochs
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=1,   # batch size for evaluation
        learning_rate=1e-6,
        warmup_steps=100,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        # load the best model when finished training (default metric is loss)
        load_best_model_at_end=True,
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=327,               # log & save weights each logging_steps
        save_steps=327,
        evaluation_strategy="steps",     # evaluate each `logging_steps`
    )

    optimizer = Adafactor(model.parameters(), scale_parameter=True,
                          relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        optimizers=(optimizer, lr_scheduler),
        # the callback that computes metrics of interest
        compute_metrics=compute_metrics,
    )

    # trainer = Trainer(
    #     model=model,                         # the instantiated Transformers model to be trained
    #     args=training_args,                  # training arguments, defined above
    #     train_dataset=train_dataset,         # training dataset
    #     eval_dataset=valid_dataset,          # evaluation dataset
    #     # the callback that computes metrics of interest
    #     compute_metrics=compute_metrics,
    # )

    # train the model
    trainer.train()

    # evaluate the current model after training
    trainer.evaluate()

    # saving the fine tuned model & tokenizer
    model_path = './../codebert-checkpoint/best-model'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
