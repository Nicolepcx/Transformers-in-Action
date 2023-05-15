import pandas as pd
import numpy as np
import torch
import evaluate
from tqdm import tqdm
from textwrap import TextWrapper
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset


class SummarizationMetrics:
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.google_bleu = evaluate.load('google_bleu')

    def compute_rouge_metrics(self, summaries, reference):
        records = []
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        for model_name in tqdm(summaries):
            predictions = summaries[model_name]
            if model_name != "TextRank (Baseline)":
                predictions = [predictions]
            references = [reference]
            results = self.rouge.compute(predictions=predictions, references=references)
            records.append(results)
        metrics_df = pd.DataFrame.from_records(records, index=summaries.keys())

        return metrics_df

    def compute_google_bleu_metrics(self, summaries, reference):
        records = []

        for model_name in tqdm(summaries):
            predictions = summaries[model_name]
            if model_name != "TextRank (Baseline)":
                predictions = [predictions]
            references  = [[reference]]
            results = self.google_bleu.compute(predictions=predictions, references=references)
            records.append(results)
        metrics_df = pd.DataFrame.from_records(records, index=summaries.keys())

        return metrics_df

    def compute_sum_metric(self, summaries, reference):
        metrics_df = self.compute_rouge_metrics(summaries, reference)
        bleu_df = self.compute_google_bleu_metrics(summaries, reference)

        metrics_df['google_bleu'] = bleu_df['google_bleu']

        return metrics_df

def compute_rouge(df, model_names):
    """
    Compute the ROUGE metric for each model's output against the "summary" column in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the text and summary columns.
        model_names (list): A list of model names to compute the ROUGE metric for.

    Returns:
        dict: A dictionary containing the ROUGE metric scores for each model.
    """
    # Initialize the rouge metric scorer
    rouge = evaluate.load('rouge')

    # Convert the summary column to a list of references
    references = df["summary"].tolist()

    # Compute the ROUGE metric for each model's output
    results = {}
    for model_name in model_names:
        predictions = df[f"{model_name}_avg"].tolist()
        scores = rouge.compute(predictions=predictions, references=references)
        results[model_name] = scores

    return results

def compute_bleu(df, model_names):
    """
    Compute the ROUGE metric for each model's output against the "summary" column in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the text and summary columns.
        model_names (list): A list of model names to compute the ROUGE metric for.

    Returns:
        dict: A dictionary containing the ROUGE metric scores for each model.
    """
    # Initialize the bleu metric scorer
    bleu= evaluate.load('google_bleu')

    # Convert the summary column to a list of references
    references = df["summary"].tolist()

    # Compute the ROUGE metric for each model's output
    results = {}
    for model_name in model_names:
        predictions = df[f"{model_name}_avg"].tolist()
        scores = bleu.compute(predictions=predictions, references=references)
        results[model_name] = scores

    return results


def print_wrapper(print):
    """Adapted from: https://stackoverflow.com/questions/27621655/how-to-overload-print-function-to-expand-its-functionality/27621927"""

    def function_wrapper(text):
        if not isinstance(text, str):
            text = str(text)
        wrapper = TextWrapper()
        return print("\n".join([wrapper.fill(line) for line in text.split("\n")]))

    return function_wrapper

print = print_wrapper(print)


def print_summaries(summaries, reference):
    print("\033[1mGround truth\033[0m")
    print(reference)
    print("")
    print("====" * 17)
    print("")
    for model_name in summaries:
        print("\033[1m" + model_name + "\033[0m")
        print(summaries[model_name])
        print("")
        print("----" * 17)
        print("")


def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_evaluation_df_and_plot(true_labels, pred_labels, target_names, model_name):
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=target_names, output_dict=True)

    # Create a dictionary to store the evaluation metrics
    evaluation_results = {
        "Model": [model_name],
        "Accuracy": [accuracy_score(true_labels, pred_labels)],
        "F1 Score": [f1_score(true_labels, pred_labels, average='weighted')],
    }

    # Add F1 scores for each class to the dictionary
    for idx, name in enumerate(target_names):
        evaluation_results[name + " F1"] = [report[name]["f1-score"]]

    # Convert the dictionary to a DataFrame
    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.set_index("Model", inplace=True)

    # Print the evaluation DataFrame and classification report
    print(evaluation_df)
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=target_names))

    # Set up the figure for plotting
    fig, ax = plt.subplots()

    # Plot the confusion matrix
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{model_name}\nAccuracy: {evaluation_results['Accuracy'][0]:.2f}\nF1 Score: {evaluation_results['F1 Score'][0]:.2f}")
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    # Add x and y axis labels
    ax.set_xlabel("Predicted Class", fontsize=18)
    ax.set_ylabel("True Class", fontsize=18)

    # Display F1 scores in the confusion matrix
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            text_color = "white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
            ax.text(j, i, f"{conf_matrix[i, j]}\n({report[target_names[i]]['f1-score']:.2f})",
                    ha="center", va="center", color=text_color, fontsize=12)

    plt.show()

    return evaluation_df


def evaluate_model(model, tokenizer, test_set, target_names, model_name="", is_dataframe=True):
    # Prepare the data
    if is_dataframe:
        test_true_labels = test_set['label'].tolist()
        test_texts = test_set['sentence'].tolist()
    else:
        test_true_labels = test_set['label']
        test_texts = test_set['text']

    # Tokenize the test data
    test_inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)

    # Get predictions
    with torch.no_grad():
        logits = model(**test_inputs).logits
    test_pred_labels = np.argmax(logits.numpy(), axis=1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_true_labels, test_pred_labels)
    f1 = f1_score(test_true_labels, test_pred_labels, average='weighted')
    conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
    report = classification_report(test_true_labels, test_pred_labels, target_names=target_names, output_dict=True, digits=4)

    # Create a dictionary to store the evaluation metrics
    evaluation_results = {
        "Model": [model_name],
        "Accuracy": [accuracy],
        "F1 Score": [f1],
    }

    # Add F1 scores for each class to the dictionary
    for idx, name in enumerate(target_names):
        evaluation_results[name + " F1"] = [report[name]["f1-score"]]

    # Convert the dictionary to a DataFrame
    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.set_index("Model", inplace=True)

    # Print the evaluation DataFrame and classification report
    print(evaluation_df)
    print("Classification Report:")
    print(classification_report(test_true_labels, test_pred_labels, target_names=target_names, digits=4))

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{model_name}\nAccuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}")
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    # Add x and y axis labels
    ax.set_xlabel("Predicted Class", fontsize=16)
    ax.set_ylabel("True Class", fontsize=16)

    # Display F1 scores in the confusion matrix
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            text_color = "white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
            ax.text(j, i, f"{conf_matrix[i, j]}\n({report[target_names[i]]['f1-score']:.4f})",
                    ha="center", va="center", color=text_color, fontsize=12)

    plt.show()

    return evaluation_df


class TextClassifier:
    """
    A class to evaluate multiple models for text classification tasks.

    Attributes
    ----------
    test_set : DataFrame
        The test dataset which is used for evaluating the models. It should include 'text' and 'label' columns.
    models : dict
        A dictionary where keys are model names and values are the model checkpoint paths.
    target_names : list
        A list of class names corresponding to the labels in the 'label' column of the test dataset.
    num_examples : int, optional
        The number of examples to use from the test set for evaluation. Default is 100.
    seed_value : int, optional
        The seed value for random operations in numpy and torch. Default is 0.
    label_mapping : dict, optional
        A dictionary mapping original labels to new ones. If not provided, default is a mapping from range(len(target_names)) to itself.

    Methods
    -------
    set_seed(seed_value=42)
        Sets the seed for numpy, torch, and cudnn to ensure results are reproducible.
    evaluate_models(num_columns=3, figsize=(15, 10))
        Evaluates all models on the test set and provides a summary of the results.
        This includes accuracy, F1 score, and a confusion matrix for each model.
        The results are also plotted for a visual comparison between models.
    """

    def __init__(self, test_set, models, target_names, num_examples=100, seed_value=0, label_mapping=None):
        self.set_seed(seed_value)
        self.test_set = test_set
        self.models = models
        self.target_names = target_names
        self.num_examples = num_examples
        self.label_mapping = label_mapping or {i: i for i in range(len(target_names))}

    @staticmethod
    def set_seed(seed_value=42):
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def evaluate_models(self, num_columns=3, figsize=(15, 10)):
        evaluation_results = []

        num_models = len(self.models)
        num_rows = (num_models + num_columns - 1) // num_columns

        fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize, sharey=True)
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        check = u'\u2705'

        for idx, (model_name, model_checkpoint) in enumerate(self.models.items()):
            print(f"{check} Evaluating {model_name}...")

            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(self.target_names))

            test_texts = self.test_set["text"][:self.num_examples]
            test_true_labels = [self.label_mapping[x] for x in self.test_set["label"][:self.num_examples]]

            test_inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                logits = model(**test_inputs).logits
            test_pred_labels = np.argmax(logits.numpy(), axis=1)

            accuracy = accuracy_score(test_true_labels, test_pred_labels)
            f1 = f1_score(test_true_labels, test_pred_labels, average='weighted')
            report = classification_report(test_true_labels, test_pred_labels, target_names=self.target_names, output_dict=True, digits=4)

            evaluation_results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "F1 Score": f1,
            })

            for i, name in enumerate(self.target_names):
                evaluation_results[-1][name + " F1"] = report[name]["f1-score"]

            ax = axes[idx // num_columns, idx % num_columns]
            conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
            im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f"{model_name}\nAccuracy: {accuracy:.2f}\nF1 Score: {f1:.2f}")
            tick_marks = np.arange(len(self.target_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(self.target_names, rotation=45, ha='right')
            ax.set_yticklabels(self.target_names)

            # Display F1 scores in the confusion matrix
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    text_color = "white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
                    ax.text(j, i, f"{conf_matrix[i, j]}\n({report[self.target_names[i]]['f1-score']:.2f})",
                            ha="center", va="center", color=text_color, fontsize=8)

        for idx in range(num_models, num_rows * num_columns):
            axes[idx // num_columns, idx % num_columns].axis("off")

        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df.set_index("Model", inplace=True)

        print(evaluation_df)
        plt.show()
        return evaluation_df


def reduce_dataset_size_and_split(dataset, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1, seed=42):
    assert train_fraction + val_fraction + test_fraction <= 1, "The sum of the fractions should not exceed 1."

    np.random.seed(seed)
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_size = int(train_fraction * num_samples)
    val_size = int(val_fraction * num_samples)
    test_size = int(test_fraction * num_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]

    train_data = dataset.select(train_indices)
    val_data = dataset.select(val_indices)
    test_data = dataset.select(test_indices)

    return train_data, val_data, test_data


class TextClassificationDataset(Dataset):
    """
    This class is designed to transform text data into a format that is suitable for training transformer-based models,
    such as BERT. It inherits from PyTorch's Dataset class, making it compatible with PyTorch's DataLoader for efficient
    data loading.

    Attributes:
    data : DataFrame
    The dataset containing the text and their corresponding labels.
    tokenizer : transformers.PreTrainedTokenizer
    The tokenizer corresponding to the transformer model to be used.
    It will be used to convert text into tokens that the model can understand.
    max_length : int
    The maximum length of the sequences. Longer sequences will be truncated, and shorter ones will be padded.

    Methods:
    len()
    Returns the number of examples in the dataset.
    getitem(index)
    Transforms the text at the given index in the dataset into a format suitable for transformer models.
    It returns a dictionary containing the input_ids, attention_mask, and the label for the given text.

    """
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'][index]
        label = self.data['label'][index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
