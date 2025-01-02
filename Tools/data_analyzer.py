import numpy as np
from Tools.pca_analyzer import PCAAnalyzer
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

    def process_pca(self):
        pca_analyzer = PCAAnalyzer(self.raw_data)
        pca_analyzer.scale_data()
        pca_analyzer.perform_pca(self.components_count)

        if PRINT_STATS:
            pca_analyzer.print_explained_variance()
            pca_analyzer.print_components()

        if SHOW_GRAPH:
            pca_analyzer.visualize_pca()

    def process_elbow(self):
        pass

    def process_kmens(self):
        pass

    def analyze(self):
        self.process_pca()