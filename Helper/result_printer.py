from config import Config
class ResultPrinter:
    def print_results(df,spm,predictions,actual, metrics):

        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

        last_data = df.tail(Config.SEQUENCE_LENGTH)
        next_day_prediction = spm.predict(last_data)
        print(f"Current closing price: ${float(df['Close'].iloc[-1]):.2f}")
        print(f"Predicted next day closing price: ${float(next_day_prediction):.2f}")

        # Print last 5 predictions vs actual values
        print("\nLast 5 predictions vs actual values:")
        for i in range(-5, 0):
            print(f"Actual: ${actual[i][0]:.2f} | Predicted: ${predictions[i][0]:.2f} | " 
                f"Difference: ${abs(actual[i][0] - predictions[i][0]):.2f} "
                f"({abs(actual[i][0] - predictions[i][0])/actual[i][0]*100:.2f}%)")