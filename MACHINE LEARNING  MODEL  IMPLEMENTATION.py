import pandas as pd
import numpy as np
import nltk
import tkinter as tk
from tkinter import scrolledtext
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download necessary nltk package
nltk.download('punkt')

# Create a synthetic dataset
data = {
    'message': [
        "Win a brand new car! Call now!",
        "Hey, are we still meeting for lunch?",
        "Congratulations! You have won $1000. Claim now.",
        "This is not spam, just checking in.",
        "Get a free vacation package! Limited time offer.",
        "Let's schedule a call for tomorrow.",
        "Exclusive deal for you! Buy now and save 50%.",
        "See you at the meeting in the afternoon.",
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Ham
}

df = pd.DataFrame(data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Convert text to numerical features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Function to display results in a new window
def show_results():
    result_window = tk.Tk()
    result_window.title("Spam Classification Results")

    text_area = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, width=60, height=20)
    text_area.pack(padx=10, pady=10)

    output_text = f"Accuracy: {accuracy:.2f}\n\nClassification Report:\n{report}"
    text_area.insert(tk.END, output_text)
    text_area.config(state=tk.DISABLED)

    result_window.mainloop()

# Show results in a new window
show_results()
