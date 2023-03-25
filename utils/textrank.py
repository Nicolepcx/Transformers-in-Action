import nltk
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('punkt')


class TextRank:
    """
        A class for creating a TextRank graph from a text and computing sentence ranks.

        Attributes:
        -----------
        sent_detector : nltk.tokenize.punkt.PunktSentenceTokenizer
            A sentence tokenizer for splitting text into sentences.

        c : sklearn.feature_extraction.text.CountVectorizer
            A count vectorizer for creating a bag-of-words representation of sentences.

        tfidf_transformer : sklearn.feature_extraction.text.TfidfTransformer
            A TF-IDF transformer for converting a bag-of-words matrix into a TF-IDF matrix.

        Methods:
        --------
        split_sentences(text: str) -> list:
            Split the input text into sentences.

        bag_of_words(sentences: list) -> scipy.sparse.csr_matrix:
            Create a bag-of-words representation for the input sentences.

        pagerank(G: nx.Graph, delta: float, max_iter: int, tol: float) -> np.ndarray:
            Compute the PageRank of the nodes in the graph G.

        rank(G: nx.Graph, delta: float, max_iter: int, tol: float) -> np.ndarray:
            Compute the normalized ranks of the nodes in the graph G.

        create_graph(sentences: list, delta: float) -> nx.Graph:
            Create a graph for the input sentences using the TF-IDF similarity.

        plot_graph(G: nx.Graph):
            Plot the graph G with edge weights.

        print_sentence_weights(G: nx.Graph, sentences: list):
            Print the sentence weights in the graph G.
        """
    def __init__(self):
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.c = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()

    def split_sentences(self, text):
        """
        Splits a block of text into individual sentences.

        Parameters:
            text (str): The text to split into sentences.

        Returns:
            list: A list of individual sentences extracted from the input text.
        """
        sentences = self.sent_detector.tokenize(text.strip())
        return sentences

    def bag_of_words(self, sentences):
        """
        Convert a  block of text into a sparse matrix representation of a Bag of Words.

        Parameters:
        -----------
        text (str):
            A block of strings representing the sentences.

        Returns:
        --------
        bag_words: scipy.sparse.csr_matrix
            A sparse matrix representation of a Bag of Words, where each row
            represents a sentence and each column represents a unique word in the
            corpus. The value in each cell represents the frequency of that word
            in that sentence.
        """
        bag_words = self.c.fit_transform(sentences)
        return bag_words

    def pagerank(self, G, delta=0.85, max_iter=100, tol=1e-6):
        """
        Compute the PageRank of the nodes in the graph G.

        The PageRank algorithm is an iterative method that estimates the importance
        of nodes in a graph based on the structure of the graph and the connections
        between nodes. This implementation uses the power iteration method to compute
        PageRank values.

        Parameters:
        -----------
        G : networkx.classes.graph.Graph
            The input graph.
        delta : float, optional
            The damping factor for the PageRank algorithm (default is 0.85).
        max_iter : int, optional
            The maximum number of iterations for the PageRank algorithm (default is 100).
        tol : float, optional
            The tolerance for convergence of the PageRank algorithm (default is 1e-6).

        Returns:
        --------
        ranks : numpy.ndarray
            A 1D array containing the PageRank values for each node in the graph. The
            values are proportional to the importance of the nodes in the graph.
        """

        N = len(G)
        ranks = np.ones(N) / N
        adjacency_matrix = nx.to_numpy_array(G)

        # Calculate the out-degree for each node
        out_degrees = np.sum(adjacency_matrix, axis=1)

        for _ in range(max_iter):
            # This line corresponds to the (1 - delta) / N term in both equations
            new_ranks = np.ones(N) * (1 - delta) / N
            for i in range(N):
                for j in range(N):
                    # This condition checks if there's an edge between nodes j and i
                    if adjacency_matrix[j, i] > 0:
                        # This line corresponds to the summation in both equations,
                        # with omega_ji being the adjacency_matrix[j, i] term,
                        # ranks[j] corresponding to r(V_j^(k)), and out_degrees[j]
                        # We representing the denominator in the fraction (c_j or the sum of the edge weights)
                        new_ranks[i] += delta * ranks[j] * adjacency_matrix[j, i] / out_degrees[j]
            # This line checks for convergence
            if np.linalg.norm(new_ranks - ranks) < tol:
                break
            ranks = new_ranks

        return ranks

    def rank(self, G, delta=0.85, max_iter=100, tol=1e-6):
        """
        Compute the normalized PageRank of the nodes in the graph G.

        This method computes the PageRank values for the nodes in the graph and
        normalizes them so that their sum equals 1. This can be useful for comparing
        the relative importance of nodes in the graph.

        Parameters:
        -----------
        G : networkx.classes.graph.Graph
           The input graph.
        delta : float, optional
           The damping factor for the PageRank algorithm (default is 0.85).
        max_iter : int, optional
           The maximum number of iterations for the PageRank algorithm (default is 100).
        tol : float, optional
           The tolerance for convergence of the PageRank algorithm (default is 1e-6).

        Returns:
        --------
        normalized_ranks : numpy.ndarray
           A 1D array containing the normalized PageRank values for each node in the graph.
           The values sum to 1 and are proportional to the relative importance of the nodes
           in the graph.
        """

        ranks = self.pagerank(G, delta, max_iter, tol)
        return ranks / ranks.sum()

    def create_graph(self, sentences, delta=0.85):
        """
        create_graph(sentences)

        This function creates a graph from a list of sentences using bag of words,
        tf-idf transformation and cosine similarity.

        Args:
        - sentences (list): A list of strings representing sentences.

        Returns:
        - G (networkx.classes.graph.Graph): A graph of the sentences, where nodes
              represent the sentences and edges represent their similarity.

        """
        # Convert the sentences to a bag of words matrix
        bag_of_words = self.bag_of_words(sentences)

        # Apply TF-IDF transformation to the bag of words matrix
        tfidf_matrix = self.tfidf_transformer.fit_transform(bag_of_words)

        # Compute the cosine similarity matrix
        adjacency_matrix = np.dot(tfidf_matrix, tfidf_matrix.T)
        # Create a graph from the similarity matrix
        G = nx.from_numpy_array(adjacency_matrix)

        for u, v, weight in G.edges(data=True):
            weight['weight'] = adjacency_matrix[u, v]

        return G

    def plot_graph(self, G):
        """
        Plots a NetworkX graph using a circular layout and displays it.

        Parameters:
        -----------
        G : NetworkX graph
            The graph to be plotted.

        Returns:
        --------
        None
        """
        # Remove self-loops from the graph
        G.remove_edges_from(nx.selfloop_edges(G))

        # Set the figure size
        plt.figure(figsize=(10, 10))

        # Set the positions for all nodes
        pos = nx.circular_layout(G)

        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='#58A3B3')

        # Draw the edges
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edges(G, pos, width=3, edge_color='gray')

        # Draw edge labels with weights
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

        # Draw the labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif', font_color='white')

        # Hide the axis
        plt.axis('off')

        # Display the plot
        plt.show()

    def print_sentence_weights(self, G, sentences):
        """
        Print the sentence weights in the TextRank graph.

        This method iterates through the edges of the graph G and prints the
        corresponding sentences and their similarity weight. It can be useful for
        visualizing and understanding the relationships between sentences in the graph.

        Parameters:
        -----------
        G : networkx.classes.graph.Graph
            The input graph representing sentence relationships.
        sentences : list of str
            A list of sentences corresponding to the nodes in the graph G.

        Returns:
        --------
        None
        """

        # Print the header
        print("Sentence Weights:")

        # Iterate through the edges of the graph
        for u, v, weight in G.edges(data=True):
            # Get the weight value from the edge data
            weight_value = weight['weight']

            # Print the sentences and their corresponding weight
            print(f"Sentence {u}: {sentences[u]}")
            print(f"Sentence {v}: {sentences[v]}")
            print(f"Weight: {weight_value:.4f}\n")



