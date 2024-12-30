import yfinance as yf
import pandas as pd

# Seznam tickerů pro analýzu (např. 10 nejběžnějších z S&P 500)
tickers = ["RGTI", "AAPL", "MSFT", 
           "GOOG", "AMZN", "TSLA", 
           "META", "NFLX", "NVDA", 
           "ADBE", "ORCL", "QBTS",
           "LCID", "SOUN", "PLTR",
           "ABEV", "PLUG", "F",
           "INTC", "QUBT", "NIO",
           "AMD", ]

# Data z YFinance obsahují:
#    Open: Cena akcie při otevření trhu.
#    High: Nejvyšší cena během dne.
#    Low: Nejnižší cena během dne.
#    *Close: Cena akcie při uzavření trhu.
#    *Volume: Počet zobchodovaných akcií během dne.
#    Dividends: Výplaty dividend.
#    Stock Splits: Informace o dělení akcií.
# atr s * jsou vhodné pro predikci

def fetch_and_analyze_data(tickers):
    """
    Stáhne data z YFinance pro dané tickery a vytvoří souhrnnou tabulku
    s počtem dostupných dat pro každý atribut.
    
    Args:
        tickers (list): Seznam tickerů.
    
    Returns:
        pd.DataFrame: Tabulka se souhrnem dostupných dat.
    """
    summary = []

    for ticker in tickers:
        # Stáhnout historická data pro daný ticker
        print(f"Fetching data for {ticker}...")
        try:
            data = yf.Ticker(ticker).history(period="max")
            if data.empty:
                print(f"Ticker {ticker} neobsahuje žádná data.")
                continue

            # Spočítat celkový počet záznamů a počet nenulových dat pro každý sloupec
            total_records = len(data)
            attribute_counts = data.count()  # Počet nenulových hodnot pro každý atribut

            summary.append({
                "Ticker": ticker,
                "Total Records": total_records,
                "Open": attribute_counts.get("Open", 0),
                "High": attribute_counts.get("High", 0),
                "Low": attribute_counts.get("Low", 0),
                "Close": attribute_counts.get("Close", 0),
                "Volume": attribute_counts.get("Volume", 0),
                "Dividends": attribute_counts.get("Dividends", 0),
                "Stock Splits": attribute_counts.get("Stock Splits", 0),
            })
        except Exception as e:
            print(f"Chyba při stahování dat pro ticker {ticker}: {e}")

    # Vytvořit DataFrame ze souhrnu
    summary_df = pd.DataFrame(summary)

    # Seřadit podle celkového počtu záznamů od nejvyššího k nejnižšímu
    summary_df = summary_df.sort_values(by="Total Records", ascending=False)

    return summary_df

# Zavolat funkci a zobrazit výsledek
results = fetch_and_analyze_data(tickers)

# Výpis všech výsledků
print(results)
results.to_csv("data_summary.csv", index=False)