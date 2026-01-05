# -------------------------------
# Hate Speech Detection Project
# -------------------------------

# Importing Libraries
import pandas as pd
import numpy as np
import re
import nltk
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
dataset = pd.read_csv("D:\\document\\train.csv")

# Map numeric classes to meaningful labels
dataset["labels"] = dataset["class"].map({
    0: "hate speech",
    1: "Offensive language",
    2: "No hate no offensive language"
})

# Select only relevant columns
data = dataset[["tweet", "labels"]]

# -------------------------------
# Step 2: Data Cleaning
# -------------------------------
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")

def clean_data(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r'<.*?>+', "", text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub(r'\n', " ", text)
    text = re.sub(r'\w*\d\w*', "", text)
    text = [word for word in text.split() if word not in stopwords]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

data.loc[:, "tweet"] = data["tweet"].apply(clean_data)

# -------------------------------
# Step 3: Prepare Data
# -------------------------------
x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

# -------------------------------
# Step 4: Train Model
# -------------------------------
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

# -------------------------------
# Step 5: User Options
# -------------------------------
while True:
    print("\nChoose an option:")
    print("1 - Show Accuracy")
    print("2 - Show Heatmap")
    print("3 - Predict a Sentence")
    print("4 - Exit")
    
    choice = input("Enter your choice: ")

    if choice == "1":
        print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

    elif choice == "2":
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="YlGnBu")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix Heatmap")
        plt.show()

    elif choice == "3":
        user_input = input("\nEnter a sentence to check: ")
        cleaned_input = clean_data(user_input)
        input_vector = cv.transform([cleaned_input])
        prediction = dt.predict(input_vector)
        print("Predicted Category:", prediction[0])

    elif choice == "4":
        print("Exiting program. Goodbye!")
        break

    else:
        print("Invalid choice, please try again.")
