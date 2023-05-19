import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE


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


def plot_tsne_3d(nb_pipeline, test_df, test_predictions):
    # Obtain the TF-IDF features of the test dataset
    tfidf_vectorizer = nb_pipeline.named_steps['tfidfvectorizer']
    X_test_tfidf = tfidf_vectorizer.transform(test_df["sentence"])

    # Apply t-SNE to the test data's TF-IDF features
    tsne = TSNE(n_components=3, random_state=42)
    X_test_tsne = tsne.fit_transform(X_test_tfidf.toarray())

    # Create a 3D scatter plot of t-SNE transformed test data with color-coded predicted labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the labels, their colors, and markers
    labels = ['Negative', 'Neutral', 'Positive']
    label_colors = {0: 'black', 1: '#58A3B3', 2: 'darkgrey'}
    label_markers = {0: 'o', 1: '^', 2: 's'}

    # Add the labels inside the plot
    for label, color in label_colors.items():
        mask = (test_predictions == label)
        ax.scatter(X_test_tsne[mask, 0], X_test_tsne[mask, 1], X_test_tsne[mask, 2], c=color, label=labels[label], marker=label_markers[label], edgecolors='k')

    ax.legend(title='Sentiment', loc=(0.98, 0.55))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    plt.show()

