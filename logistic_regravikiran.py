import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load data from CSV
def load_data(csv_file):
    stock_data = pd.read_csv(csv_file)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    stock_data = stock_data[['Close']]
    stock_data = stock_data.dropna()
    return stock_data


# Prepare the dataset
def prepare_data(stock_data):
    stock_data['Target'] = stock_data['Close'].shift(-30) > stock_data['Close']
    stock_data['Target'] = stock_data['Target'].astype(int)
    X = np.array(stock_data.index.map(pd.Timestamp.toordinal)).reshape(-1, 1)[:-30]
    y = np.array(stock_data['Target'])[:-30]
    return X, y


# Print class distribution
def print_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    print(f'Class distribution: {dict(zip(unique, counts))}')


# Train and evaluate the model
def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{cm}')
    print(f'Classification Report:\n{report}')
    return model


# Make future predictions
def make_predictions(stock_data, model, days=30):
    last_date = stock_data.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days + 1)]
    future_dates_ordinals = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
    future_predictions = model.predict(future_dates_ordinals)
    future_data = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted_Up'])
    return future_data


# Plot the results
def plot_results(stock_data, future_data):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'], label='Actual Close Price')
    plt.scatter(future_data.index, stock_data['Close'].iloc[-len(future_data):][future_data['Predicted_Up'] == 1],
                color='red', label='Predicted Up', marker='^')
    plt.scatter(future_data.index, stock_data['Close'].iloc[-len(future_data):][future_data['Predicted_Up'] == 0],
                color='blue', label='Predicted Down', marker='v')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    csv_file = "M&M.NS.csv"  # Replace with your CSV file path
    stock_data = load_data(csv_file)
    X, y = prepare_data(stock_data)

    # Print class distribution
    print_class_distribution(y)

    model = train_and_evaluate_model(X, y)
    future_data = make_predictions(stock_data, model, days=30)

    # Print future predictions
    print("Next 30 days predicted stock movements (1: Up, 0: Down):")
    print(future_data)

    plot_results(stock_data,future_data)