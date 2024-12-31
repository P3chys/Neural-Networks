import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class KmeansAnalyzer:
    def __init__(self, raw_data, pca_data, n_clusters = 2):
        self.data = pca_data
        self.pca = pca_data
        pca_df = pd.DataFrame(self.pca, columns=['PCA_1', 'PCA_2'], index=raw_data.index)
        self.kmeans_df = pd.concat([raw_data, pca_df], axis=1)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=10)
        self.kmeans_df['Cluster'] = self.kmeans.fit_predict(self.kmeans_df)

    def visualize(self):
        # Create Plotly figure for 2D scatter plot
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter'}]])

        # Add scatter plot for each cluster
        for cluster in range(7):
            cluster_points = self.kmeans_df[self.kmeans_df['Cluster'] == cluster]
            
            # Vytvoření textu pro hover
            hover_text = cluster_points.apply(
                lambda row: '<br>'.join(
                    [f'{col}: {row[col]}' for col in cluster_points.columns]
                ), axis=1
            )
            fig.add_trace(go.Scatter(x=cluster_points['PCA_1'], y=cluster_points['PCA_2'], mode='markers', name=f'Cluster {cluster + 1}', hoverinfo='text', text=hover_text))

        # Update layout
        fig.update_layout(title='K-Means Clustering', xaxis_title='PCA1', yaxis_title='PCA2')

        # Show plot
        fig.show()

    def print(self):
        print("K-Means:")
        print(self.kmeans_df)

if __name__ == "__main__":
    from pca_analyzer import PCAAnalyzer  # Odkaz na vaši vlastní PCA třídu
    from config import Config
    import yfinance as yf

    ticker = "AAPL"  # Zadejte vlastní ticker
    raw_data = yf.download(ticker, start="2010-01-01", end="2023-12-31")
    selected_columns = ["Open", "High", "Low", "Close", "Volume"]
    raw_data_selected = raw_data[selected_columns]

    # Inicializace a spuštění PCA
    components_count = Config.PCA_NUM
    pca_analyzer = PCAAnalyzer(raw_data_selected)
    pca_analyzer.scale_data()
    pca_analyzer.perform_pca(components_count)
    pca_data = pca_analyzer.pca_result

    # Provedení K-Means klasterizace
    analyzer = KmeansAnalyzer(raw_data, pca_data, components_count)
    analyzer.print()
    analyzer.visualize()