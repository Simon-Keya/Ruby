from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Create and train the classifier
def train_classifier(X, y):
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
    ])
    classifier.fit(X, y)
    return classifier

# Save the trained classifier to a file
def save_classifier(classifier, spam_classifier_model):
    joblib.dump(classifier, spam_classifier_model.pkl)
