from flask import Flask, render_template, request
import joblib
from src.preprocess import clean_text


app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    
    if request.method == "POST":
        text = request.form["text"]
        cleaned_text = clean_text(text)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

