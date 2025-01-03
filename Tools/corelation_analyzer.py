import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

class CorrelationAnalyzer:
    def __init__(self, data: pd.DataFrame):

        self.data = data
        self.correlation_matrix = None

    def calculate_correlation(self, method: str = 'pearson'):
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
        return pd.DataFrame(self.correlation_matrix)

    def plot_correlation_matrix(self):
        plt.figure(figsize=(15, 20))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

    def save_correlation_matrix(self, path = "korelace.png"):
        plt.figure(figsize=(25, 15))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.savefig(path, dpi=300, bbox_inches='tight') 

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
