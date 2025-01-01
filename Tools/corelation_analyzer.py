import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

class CorrelationAnalyzer:
    def __init__(self, data: pd.DataFrame):
        """
        Inicializace CorrelationAnalyzer s daty.

        Args:
        data: Vstupní DataFrame s daty, nad kterými bude prováděna analýza korelace.
        """
        self.data = data
        self.correlation_matrix = None

    def calculate_correlation(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Vypočítá korelační matici.

        Args:
        method: pearson, kendall, spearman; Default: pearson.

        Returns:
        pd.DataFrame: Korelační matice.
        """
        if method not in ['pearson', 'kendall', 'spearman']:
            raise ValueError("Method must be 'pearson', 'kendall', or 'spearman'.")
        
        self.correlation_matrix = self.data.corr(method=method)
        return self.correlation_matrix

    def plot_correlation_matrix(self, figsize: tuple = (10, 8), annot: bool = True, cmap: str = 'coolwarm'):
        """
        Vykreslí korelační matici pomocí heatmapy.

        Args:
            figsize (tuple): Velikost obrázku (šířka, výška).
            annot (bool): Zda anotovat hodnoty na heatmapě.
            cmap (str): Barevná mapa pro heatmapu.
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix has not been calculated. Call `calculate_correlation` first.")
        
        plt.figure(figsize=figsize)
        sns.heatmap(self.correlation_matrix, annot=annot, cmap=cmap, fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

# Příklad použití
if __name__ == "__main__":
    # Definice tickeru a načtení dat
    ticker = "AAPL"  # Zadejte vlastní ticker
    data = yf.Ticker(ticker).history(start="2010-01-01", end="2023-12-31")
    
    # Vybereme relevantní sloupce pro PCA
    selected_columns = ["Open", "High", "Low", "Close", "Volume"]
    #data = data[selected_columns]

    analyzer = CorrelationAnalyzer(data)
    print("Korelacni matice:")
    print(analyzer.calculate_correlation())

    print("\nZobrazeni korelacni matice:")
    analyzer.plot_correlation_matrix()
