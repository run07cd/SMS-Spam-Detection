import io
import urllib.request
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_PATH = Path("data/SMSSpamCollection")


def load_messages():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading SMS Spam Collection dataset...")
        with urllib.request.urlopen(DATA_URL, timeout=30) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
            DATA_PATH.write_bytes(archive.read("SMSSpamCollection"))

    messages = []
    labels = []
    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            label, message = line.rstrip("\n").split("\t", 1)
            labels.append(label)
            messages.append(message)
    return messages, labels


def main():
    messages, labels = load_messages()
    X_train, X_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB(),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print(classification_report(y_test, predictions, zero_division=0))

    examples = [
        "Congratulations! You have won a free prize. Claim now.",
        "Hi, are we still meeting after class today?",
    ]
    for message, prediction in zip(examples, model.predict(examples)):
        print(f"\nMessage: {message}")
        print("Prediction:", prediction)


if __name__ == "__main__":
    main()
