from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# model এবং vectorizer load
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text")

        if text == "":
            return render_template("index.html", result="কোনো লেখা দেওয়া হয়নি")

        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]

        return render_template("index.html", result=prediction)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
