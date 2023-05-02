import matplotlib.pyplot as plt
import pandas as pd

def plot_dataset_distribution(datasets, dataset_names, label_map=None):
    assert label_map is not None, "The label_map cannot be None"
    if label_map is None:
        raise ValueError("label_map should be provided.")

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)

    if n == 1:
        axes = [axes]

    for dataset, dataset_name, ax in zip(datasets, dataset_names, axes):
        # get the label names and their counts
        if isinstance(dataset, pd.DataFrame):
            label_counts = dataset["label"].map(label_map).value_counts()
            label_names = label_counts.index.tolist()
        else:
            label_counts = dataset.map(lambda x: label_map[x['label']]).count_by_column("label")
            label_names = list(label_counts.keys())

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
