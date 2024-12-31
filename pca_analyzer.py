import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt
from config import Config

class PCAAnalyzer:
    def __init__(self, data: pd.DataFrame):
        """
        Inicializuje PCAAnalyzer s poskytnutými daty.

        Args:
            data: DataFrame obsahující data pro analýzu.
        """
        self.data = data
        self.scaled_data = None
        self.pca = None
        self.pca_result = None
        self.explained_variance = None
        self.components = None
    
    def scale_data(self):
        """Škáluje data na průměr 0 a směrodatnou odchylku 1."""
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)
    
    def perform_pca(self, n_components: int = 2):
        """
        Provádí PCA na škálovaných datech.

        Args:
            n_components: Počet hlavních komponent.
        """
        if self.scaled_data is None:
            raise ValueError("Data nejsou škálována. Nejprve zavolejte scale_data().")
        
        self.pca = PCA(n_components=n_components)
        self.pca_result = self.pca.fit_transform(self.scaled_data)
        self.explained_variance = self.pca.explained_variance_ratio_
        self.components = pd.DataFrame(
            self.pca.components_,
            columns=self.data.columns,
            index=[f"PC{i+1}" for i in range(n_components)]
        )
    
    def visualize_pca(self):
        """Vizualizuje první dvě hlavní komponenty."""
        if self.pca_result is None:
            raise ValueError("PCA nebylo provedeno. Nejprve zavolejte perform_pca().")
        
        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.pca_result[:, 0], self.pca_result[:, 1],
            c='blue', edgecolor='k', s=50
        )
        plt.title("PCA Vizualizace")
        plt.xlabel(f"PC1 ({self.explained_variance[0]:.2%} variability)")
        plt.ylabel(f"PC2 ({self.explained_variance[1]:.2%} variability)")
        plt.grid()
        plt.show()
    
    def print_explained_variance(self):
        """Tiskne vysvetlenou variabilitu jednotlivych komponent."""
        if self.explained_variance is None:
            raise ValueError("PCA nebylo provedeno. Nejprve zavolejte perform_pca().")
        
        print("Vysvetlena variabilita jednotlivych komponent:")
        for i, variance in enumerate(self.explained_variance):
            print(f"PC{i+1}: {variance:.2%}")
    
    def print_components(self):
        """Tiskne vliv atributů na hlavní komponenty."""
        if self.components is None:
            raise ValueError("PCA nebylo provedeno. Nejprve zavolejte perform_pca().")
        
        print("Vliv atributu na hlavni komponenty:")
        print(self.components)

if __name__ == "__main__":
    # Definice tickeru a načtení dat
    ticker = "AAPL"  # Zadejte vlastní ticker
    data = yf.download(ticker, start="2010-01-01", end="2023-12-31")
    
    # Vybereme relevantní sloupce pro PCA
    selected_columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data[selected_columns]
    
    # Inicializace a spuštění PCA
    print("POKUS #1")
    components_count = Config.PCA_NUM
    pca_analyzer = PCAAnalyzer(data)
    pca_analyzer.scale_data()
    pca_analyzer.perform_pca(components_count)
    pca_analyzer.print_explained_variance()
    pca_analyzer.print_components()
    pca_analyzer.visualize_pca()

    # 2. pokus
    print("POKUS #2")
    data = yf.Ticker(ticker).history(start="2010-01-01", end="2023-12-31")
    components_count = Config.PCA_NUM + 2 # původní počet nestačí
    pca_analyzer = PCAAnalyzer(data)
    pca_analyzer.scale_data()
    pca_analyzer.perform_pca(components_count)
    pca_analyzer.print_explained_variance()
    pca_analyzer.print_components()
    pca_analyzer.visualize_pca()
