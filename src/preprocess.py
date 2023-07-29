import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path, encoding="latin-1")
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    return data

# Text preprocessing
def preprocess_text(text):
    stemmer = PorterStemmer()
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stopwords.words('english')]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)
