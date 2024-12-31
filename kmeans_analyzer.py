import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class KmeansAnalyzer:
    def __init__(self, raw_data, pca_data, n_clusters = 2):
        self.data = pca_data
        self.clusters = n_clusters
        kmeans = KMeans(n_clusters=self.clusters, random_state=10)
        self.data['cluster'] = kmeans.fit_predict(self.data)

    def visualize(self):
        pass
    
    def print(self):
        # sjednotit výsledky s raw_data
        pass

if __name__ == "__main__":
    from pca_analyzer import PCAAnalyzer  # Odkaz na vaši vlastní PCA třídu
    from config import Config
    import yfinance as yf

    ticker = "AAPL"  # Zadejte vlastní ticker
    data = yf.download(ticker, start="2010-01-01", end="2023-12-31")
    selected_columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data[selected_columns]
    
    # Inicializace a spuštění PCA
    components_count = Config.PCA_NUM
    pca_analyzer = PCAAnalyzer(data)
    pca_analyzer.scale_data()
    pca_analyzer.perform_pca(components_count)
    data = pca_analyzer.pca_result
    print(data)