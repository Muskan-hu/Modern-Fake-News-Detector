from flask import Flask, render_template, request
import re
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from scipy.sparse import hstack
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# -------------------------
# Load model and vectorizer
# -------------------------
bundle = joblib.load("fake_news_detector.pkl")
vectorizer = bundle['vectorizer']
svd = bundle['svd']
ensemble = bundle['model']
USE_SVD = svd is not None

# -------------------------
# Preprocessing setup
# -------------------------
STOP_WORDS = set(stopwords.words('english'))
PS = PorterStemmer()
RE_NON_LETTERS = re.compile(r'[^a-zA-Z\s]')
RE_MULTI_WS = re.compile(r'\s+')

def preprocess_and_stem(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text else ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = RE_NON_LETTERS.sub(' ', text)
    text = RE_MULTI_WS.sub(' ', text).strip()
    words = text.split()
    stemmed = [PS.stem(w) for w in words if w not in STOP_WORDS]
    return " ".join(stemmed)

# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form.get('news_text', '').strip()
    if not news_text:
        return render_template('index.html', error="Please enter some text!")

    # Preprocess and features
    news_proc = preprocess_and_stem(news_text)
    xv = vectorizer.transform([news_proc])
    sentiment_score = TextBlob(news_proc).sentiment.polarity
    text_length = len(news_proc)
    extra = np.array([[sentiment_score, text_length]])
    x_full = hstack([xv, extra])
    x_final = svd.transform(x_full) if USE_SVD else x_full

    # Predict
    pred = ensemble.predict(x_final)
    pred_proba = ensemble.predict_proba(x_final)

    if pred[0] == 0:  # Fake news
        return render_template('result_fake.html', 
                               fake_prob=f"{pred_proba[0][0]*100:.2f}%",
                               real_prob=f"{pred_proba[0][1]*100:.2f}%")
    else:  # Real news
        return render_template('result_real.html', 
                               fake_prob=f"{pred_proba[0][0]*100:.2f}%",
                               real_prob=f"{pred_proba[0][1]*100:.2f}%")

# -------------------------
# Main entry point
# -------------------------
if __name__ == '__main__':
    # For local testing only (Render will use gunicorn)
    app.run(host='0.0.0.0', port=5000, debug=True)
