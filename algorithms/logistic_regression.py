import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

def toxic_comment_classifier(train_path, test_path, class_names, word_ngram_range=(1, 1), char_ngram_range=(2, 6),
                             word_features=10000, char_features=50000, C=0.1, solver='sag', cv_folds=3):
    """
    Trains a logistic regression model for multi-label text classification (toxic comment detection).
    
    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the test CSV file.
        class_names (list): List of class names (target columns).
        word_ngram_range (tuple): N-gram range for word-based TF-IDF vectorisation.
        char_ngram_range (tuple): N-gram range for character-based TF-IDF vectorisation.
        word_features (int): Maximum number of features for word-based TF-IDF.
        char_features (int): Maximum number of features for character-based TF-IDF.
        C (float): Inverse regularisation strength for logistic regression.
        solver (str): Optimisation algorithm for logistic regression.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        pd.DataFrame: Submission dataframe with predicted probabilities for each class.
        list: Cross-validation scores for each class.
    """
    # Load and preprocess data
    train = pd.read_csv(train_path).fillna(' ')
    test = pd.read_csv(test_path).fillna(' ')
    train_text = train['comment_text']
    test_text = test['comment_text']
    all_text = pd.concat([train_text, test_text])

    # Word-based TF-IDF vectorisation
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=word_ngram_range,
        max_features=word_features)
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)

    # Character-based TF-IDF vectorisation
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=char_ngram_range,
        max_features=char_features)
    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)

    # Combine features
    train_features = hstack([train_char_features, train_word_features])
    test_features = hstack([test_char_features, test_word_features])

    # Model training and prediction
    scores = []
    submission = pd.DataFrame({'id': test['id']})
    for class_name in class_names:
        train_target = train[class_name]
        classifier = LogisticRegression(C=C, solver=solver)

        # Cross-validation score
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=cv_folds, scoring='roc_auc'))
        scores.append(cv_score)
        print(f'CV score for class {class_name} is {cv_score}')

        # Fit model and predict probabilities
        classifier.fit(train_features, train_target)
        submission[class_name] = classifier.predict_proba(test_features)[:, 1]

    return submission, scores

# Example usage
if __name__ == "__main__":
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_path = "datasets/toxic_comment_input/train.csv"
    test_path = "datasets/toxic_comment_input/test.csv"

    submission, scores = toxic_comment_classifier(
        train_path=train_path,
        test_path=test_path,
        class_names=class_names,
        word_ngram_range=(1, 1),
        char_ngram_range=(2, 6),
        word_features=10000,
        char_features=50000,
        C=0.1,
        solver='sag',
        cv_folds=3
    )
    submission.to_csv("toxic_comment_submission.csv", index=False)
