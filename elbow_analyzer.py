import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ElbowAnalyzer:
    def __init__(self, data: np.ndarray):
        """
        Inicializuje ElbowAnalyzer s vysvětlenou variancí.
        
        Args:
            explained_variance (np.ndarray): Pole vysvětlené variance z PCA.
        """
        self.data = data
        self.wcss = [] # WCSS = Within-Cluster Sum of Squares (součet čtverců uvnitř klastrů)
        # Pro každý bod v klastru se vypočítá vzdálenost mezi bodem a středem klastru (centroidem).
        # Tyto vzdálenosti se umocní na druhou (čtverce) a sečtou.
        # Tento proces se opakuje pro každý klastr a výsledné hodnoty se sečtou.
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 10)
            kmeans.fit(self.data)
            self.wcss.append(kmeans.inertia_)
    
    def plot_elbow_curve(self):
        """
        Vykreslí Elbow křivku na základě kumulativní vysvětlené variance.
        """
        plt.plot(range(1, 11), self.wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

# Testovací kód, pokud se spustí tento soubor samostatně
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
    
    # Inicializace ElbowAnalyzer s vysvětlenou variancí
    analyzer = ElbowAnalyzer(data)
    
    # Vykreslení Elbow křivky
    analyzer.plot_elbow_curve()