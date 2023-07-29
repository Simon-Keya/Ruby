from src.preprocess import load_dataset, preprocess_text, train_test_split
from src.train_classifier import train_classifier, save_classifier

# Step 3: Load and preprocess the dataset
data = load_dataset('dataset.csv')
data['text'] = data['text'].apply(preprocess_text)

# Step 5: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Step 6: Build and train the classifier
classifier = train_classifier(X_train, y_train)

# Step 7: Evaluate the model
# ... (code for evaluating the model and making predictions)

# Step 10: Save the model
save_classifier(classifier, 'spam_classifier_model.pkl')
