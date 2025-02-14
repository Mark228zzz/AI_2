from sklearn.naive_bayes import MultinomialNB
from trainer import show_data_info, split_data, train_model, test_model, test_unseen_data

def main():
    model = MultinomialNB(alpha=0.1) # Set alpha 0.1

    df = show_data_info()

    X_train, X_test, y_train, y_test, tfidf = split_data(df)

    train_model(model, X_train, y_train)

    test_model(model, X_test, y_test)

    test_unseen_data(model, tfidf)

if __name__ == '__main__':
    main()
