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
            references = [reference]
            results = self.rouge.compute(predictions=predictions, references=references)
            records.append(results)
        metrics_df = pd.DataFrame.from_records(records, index=summaries.keys())

        return metrics_df

    def compute_google_bleu_metrics(self, summaries, reference):
        records = []

        for model_name in tqdm(summaries):
            predictions = summaries[model_name]
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

def print_wrapper(print):
    """Adapted from: https://stackoverflow.com/questions/27621655/how-to-overload-print-function-to-expand-its-functionality/27621927"""

    def function_wrapper(text):
        if not isinstance(text, str):
            text = str(text)
        wrapper = TextWrapper()
        return print("\n".join([wrapper.fill(line) for line in text.split("\n")]))

    return function_wrapper

print = print_wrapper(print)
