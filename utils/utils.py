import pandas as pd
import evaluate
from tqdm import tqdm
from textwrap import TextWrapper

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
