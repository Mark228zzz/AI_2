import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def show_data_info():
    df = pd.read_csv('~/.cache/kagglehub/datasets/uciml/sms-spam-collection-dataset/versions/1/spam.csv', encoding='latin-1')

    # Leave only v1 and v2 and replace with label, message
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    print(df['label'].value_counts())

    ham, spam = list(df['label'].value_counts())

    # Print the ratio of ham and spam in dataset
    print(f'Ham/Spam Ratio: [Ham: {ham/(ham+spam)*100:.2f}% | Spam: {spam/(ham+spam)*100:.2f}%]')

    # Plot a graph of ratio
    df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.show()

    # Plot common words in spams
    spam_words = ' '.join(df[df['label'] == 'spam']['message'])
    wordcloud = WordCloud(width=1024, height=1024).generate(spam_words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('Spam Messages')
    plt.show()

    # Plot common words in hams
    spam_words = ' '.join(df[df['label'] == 'ham']['message'])
    wordcloud = WordCloud(width=1024, height=1024).generate(spam_words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('Ham Messages')
    plt.show()

    return df

def split_data(df):
    labels = df['label'].map({'ham': 0, 'spam': 1}).values # Transform into number 0, 1 and get an numpy ndarray
    data = df['message'].values # Get a numpy ndarray

    # 20% for test, %80 for train
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    tfidf = TfidfVectorizer(max_features=5_000) # Create a tfidf
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    return X_train, X_test, y_train, y_test, tfidf

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train) # Train(fit) the model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test) # Get model`s predictions

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Metrics
    print('=== Metrics ===')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
    print(f'ROC-AUC: {roc_auc_score(y_test, y_pred):.3f}\n')

    print('=== Classification Report ===')
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Plot a confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def test_unseen_data(model, tfidf):
    # Unseen test data for spam and ham emails
    test_spam = [
        'Congrats! You got a free prize! Click on this link to claim the reward: http://scam.scam',
        'F1ee mone y y0u w0n $1,ooo,ooo to geL this cllck heIe: http://get_money.scam',
        'Your PayPal account has been locked! Confirm details here: http://fake-paypal-login.com',
        'Amazon Alert: Unusual activity detected. Verify now: http://amazon-security-scam.net',
        'FREE iPhone 15 Pro! Click to claim: http://freegad-offer.cc',
        'I need your help transferring $10M USD. You will be payed. Reply for details. - Dr. Abacha',
        'We have footage of you. Pay $5000 BTC or we leak it: http://blackmail-threat-web.site',
        'Inheritance notification: Congrats! You are owed $7.5 million FOR FREE. Contact: attor-scam@fake-lam.com'
    ]

    test_ham = [
        'Hey, are we still meeting at 7pm tonight?',
        'Your doctor`s appointment is confirmed for tomorrow at 3pm.',
        'I`ll pick up groceries on the way home. Need anything?',
        'The project deadline has been extended to Friday.',
        'Thanks for the notes from yesterday`s meeting!',
        'Can you send me the recipe for that pasta dish?',
        'Mom called and said she`ll visit next weekend.',
        'The team lunch is at the new Italian place downtown.'
    ]

    # Transform emails using TF-IDF vectorizer
    spam_emails = tfidf.transform(test_spam)
    ham_emails = tfidf.transform(test_ham)

    # Get prediction from the model
    # Assumption: model.predict returns 0 for ham and 1 for spam
    spam_pred = model.predict(spam_emails)
    ham_pred = model.predict(ham_emails)

    # Define true labels
    y_true_spam = np.ones(len(test_spam), dtype=int)  # Spam emails labeled as 1
    y_true_ham = np.zeros(len(test_ham), dtype=int) # Ham emails labeled as 0

    # Calculate accuracies
    spam_accuracy = np.mean(spam_pred == y_true_spam) * 100
    ham_accuracy = np.mean(ham_pred == y_true_ham) * 100

    # Display predictions and accuracies
    print('=== Spam Email Predictions ===')
    print(f'Predicted Labels: {spam_pred}')
    print(f'Spam Accuracy: {spam_accuracy:.2f}%\n')

    print('=== Ham Email Predictions ===')
    print(f'Predicted Labels: {ham_pred}')
    print(f'Ham Accuracy: {ham_accuracy:.2f}%\n')

    # Combine results for overall metrics
    y_true = np.concatenate([y_true_spam, y_true_ham])
    y_pred = np.concatenate([spam_pred, ham_pred])

    print('=== Classification Report ===')
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))

    # Create a confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualize the confusion matrix using a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
