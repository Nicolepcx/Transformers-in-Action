import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

nltk.download('punkt')

class TextRank:
    """
    A class for creating a TextRank graph from a list of sentences.

    Attributes:
    -----------
    sentences : list
        A list of strings representing the sentences.

    graph : networkx.classes.graph.Graph
        A graph of the sentences, where nodes represent the sentences and edges
        represent their similarity.

    Methods:
    --------
    create_graph():
        Create a graph from the list of sentences using bag of words, TF-IDF
        transformation and cosine similarity.

    plot_graph():
        Plot the TextRank graph using a circular layout and display it.

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
        Convert a list of sentences into a sparse matrix representation of a Bag of Words.

        Parameters:
        -----------
        sentences: list
            A list of strings representing the sentences.

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

    def create_graph(self, sentences):
        '''
        create_graph(sentences)

        This function creates a graph from a list of sentences using bag of words,
        tf-idf transformation and cosine similarity.

        Args:
        - sentences (list): A list of strings representing sentences.

        Returns:
        - G (networkx.classes.graph.Graph): A graph of the sentences, where nodes
              represent the sentences and edges represent their similarity.

        '''
        # Convert the sentences to a bag of words matrix
        bag_of_words = self.bag_of_words(sentences)

        # Apply TF-IDF transformation to the bag of words matrix
        tfidf_matrix = self.tfidf_transformer.fit_transform(bag_of_words)

        # Compute the cosine similarity matrix
        cosine_similarity = np.dot(tfidf_matrix, tfidf_matrix.T)

        # Create a graph from the similarity matrix
        G = nx.from_scipy_sparse_array(cosine_similarity)

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
        nx.draw_networkx_edges(G, pos, width=3, edge_color='gray')

        # Draw the labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif', font_color='white')

        # Hide the axis
        plt.axis('off')

        # Display the plot
        plt.show()
