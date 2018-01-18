"""
Module containing model fitting code for a web application that implements a
text classification model.

When run as a module, this will load a csv dataset, train a classification
model, and then pickle the resulting model object to disk.
"""
import cPickle as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class TextClassifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit a naive bayes model to the resulting features.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._classifier = MultinomialNB()

    def fit(self, X, y):
        """Fit a text classifier model.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        # Code to fit the model.
        X_vec = self._vectorizer.fit_transform(X)
        self._classifier.fit(X_vec,y)
        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        X_vec = self._vectorizer.transform(X)
        return self._classifier.predict_proba(X_vec)
    def predict(self, X):
        """Make predictions on new data."""
        X_vec = self._vectorizer.transform(X)
        return self._classifier.predict(X_vec)

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        X_vec = self._vectorizer.transform(X)
        return self._classifier.score(X_vec, y)

def get_data(filename):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.

    Returns
    -------
    X: A numpy array containing the text fragments used for training.
    y: A numpy array containing labels, used for model response.
    """
    table = pd.read_csv(filename)
    return table["body"].values, table["section_name"].values


if __name__ == '__main__':
    X, y = get_data("data/articles.csv")
    tc = TextClassifier()
    tc.fit(X, y)
    with open('model.pkl', 'w') as f:
        pickle.dump(tc, f)
