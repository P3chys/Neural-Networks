import numpy as np
from Tools.pca_analyzer import PCAAnalyzer
from Tools.elbow_analyzer import ElbowAnalyzer
from Tools.kmeans_analyzer import KmeansAnalyzer
from Tools.corelation_analyzer import CorrelationAnalyzer
from config import Config

PCA_NUM = Config.PCA_NUM
PRINT_STATS = Config.PRINT_STATS
SAVE_STATS = Config.SAVE_STATS
SAVE_GRAPH = Config.SAVE_GRAPH
SHOW_GRAPH = Config.SHOW_GRAPH
GRAPH_PATH = Config.GRAPH_PATH

class DataAnalyzer:

    def __init__(self, data: np.ndarray):
        self.raw_data = data
        self.components_count = PCA_NUM

    def process_corelation(self):
        analyzer = CorrelationAnalyzer(self.raw_data)
        korelace = analyzer.calculate_correlation()

        if PRINT_STATS:
            print("Korelacni matice:")
            print(korelace)

        if SHOW_GRAPH:
            analyzer.plot_correlation_matrix()

        if SAVE_GRAPH:
            analyzer.save_correlation_matrix(path=GRAPH_PATH + "korelace.png")

    def process_pca(self):
        pca_analyzer = PCAAnalyzer(self.raw_data)
        pca_analyzer.scale_data()
        pca_analyzer.perform_pca(self.components_count)
        self.pca_result = pca_analyzer.pca_result

        if PRINT_STATS:
            pca_analyzer.print_explained_variance()
            pca_analyzer.print_components()

        if SHOW_GRAPH:
            pca_analyzer.visualize_pca()

        if SAVE_GRAPH:
            pca_analyzer.save_pca_graph(path=GRAPH_PATH + "PCA.png")

    def process_elbow(self):
        analyzer = ElbowAnalyzer(self.pca_result)

        if SHOW_GRAPH:
            analyzer.plot_elbow_curve()

        if SAVE_GRAPH:
            analyzer.save_elbow_curve(path=GRAPH_PATH + "Elbow.png")

    def process_kmens(self):
        analyzer = KmeansAnalyzer(self.raw_data, self.pca_result, self.components_count)

        if PRINT_STATS:
            analyzer.print()

        if SHOW_GRAPH:
            analyzer.visualize()

        if SAVE_GRAPH:
            analyzer.save_graph(path=GRAPH_PATH + "K-Means.png")

    def analyze(self):
        self.process_corelation()
        self.process_pca()
        self.process_elbow()
        #self.process_kmens()