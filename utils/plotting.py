import matplotlib.pyplot as plt
import pandas as pd


def plot_dataset_distribution(datasets, dataset_names, label_map=None):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)

    if n == 1:
        axes = [axes]

    for dataset, dataset_name, ax in zip(datasets, dataset_names, axes):
        # Convert the dataset to a DataFrame if it's not already
        if not isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_pandas()

        label_counts = dataset["label"].map(label_map).value_counts()
        label_names = label_counts.index.tolist()

        # plot the label distribution
        colors = ['#58A3B3', '#f9f7ef', 'darkgrey', 'gray', '#ddebee', 'gainsboro']
        bars = ax.bar(range(len(label_counts.values)), list(label_counts.values), color=colors)
        ax.set_title(f"{dataset_name} ({len(dataset)} samples)")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(label_counts.values)))
        ax.set_xticklabels(label_names)

        # Display counts above each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom")

    plt.show()

