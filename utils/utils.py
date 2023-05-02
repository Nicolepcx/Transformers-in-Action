import pandas as pd
import numpy as np
import torch
import evaluate
from tqdm import tqdm
from textwrap import TextWrapper
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

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
