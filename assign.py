import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

def split_data(df):
    return train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

def vectorize_data(X_train, X_test, method='bow'):
    if method == 'bow':
        vectorizer = CountVectorizer(stop_words='english')
    else:
        vectorizer = TfidfVectorizer(stop_words='english')
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, vectorizer

def train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test, method_name):
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    print(f"\n🔎 {method_name} Classification Report")
    print(classification_report(y_test, y_pred))
    
    # Collect metrics
    return {
        'Method': method_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

def plot_metrics(metrics_list):
    methods = [m['Method'] for m in metrics_list]
    accuracy = [m['Accuracy'] for m in metrics_list]
    precision = [m['Precision'] for m in metrics_list]
    recall = [m['Recall'] for m in metrics_list]
    f1 = [m['F1 Score'] for m in metrics_list]

    bar_width = 0.2
    index = range(len(methods))

    plt.figure(figsize=(10, 6))
    plt.bar([i - 1.5*bar_width for i in index], accuracy, width=bar_width, label='Accuracy')
    plt.bar([i - 0.5*bar_width for i in index], precision, width=bar_width, label='Precision')
    plt.bar([i + 0.5*bar_width for i in index], recall, width=bar_width, label='Recall')
    plt.bar([i + 1.5*bar_width for i in index], f1, width=bar_width, label='F1 Score')

    plt.xticks(index, methods)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def main():
    df = load_data("IMDB Dataset.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    metrics = []

    X_train_bow, X_test_bow, _ = vectorize_data(X_train, X_test, method='bow')
    metrics.append(train_and_evaluate(X_train_bow, X_test_bow, y_train, y_test, "Bag-of-Words"))

    X_train_tfidf, X_test_tfidf, _ = vectorize_data(X_train, X_test, method='tfidf')
    metrics.append(train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF"))

    plot_metrics(metrics)

if __name__ == "__main__":
    main()
