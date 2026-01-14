# Sentiment Analysis Flask App

This project is an end-to-end Sentiment Analysis web application built using Python, Machine Learning, and Flask.
It analyzes user-provided text and predicts whether the sentiment is Positive or Negative in real time.

# Features

* Text preprocessing and cleaning

* TF-IDF feature extraction

* Logistic Regression machine learning model

* Flask-based web application

* Real-time sentiment prediction via browser

# Tech Stack

* Python

* Pandas

* Scikit-learn

* Flask

* HTML / CSS

# Project Structure
sentiment-analysis-flask-app/

│
├── data/

│   └── test (1).csv

│

├── src/

    ├── __init__.py
    
│   ├── preprocess.py

│   └── train_model.py

│

├── templates/

│   └── index.html

│

├── app.py

├── model.pkl

├── README.md

├── requirements.txt

├── test.ipynb

└── vectorizer.pkl

# How to Run the Project

* Install Dependencies
pip install -r requirements.txt

* Train the Model
python train_model.py


This will train the sentiment analysis model and save:

Trained Logistic Regression model

TF-IDF vectorizer

* Run the Flask App
python app.py

* Open in Browser
http://127.0.0.1:5000/

# Use Case

* Customer feedback analysis

* Product review sentiment detection

* Social media text analysis

* Beginner-friendly NLP & Flask project

# Author

Anugraha AL

B.Sc Physics Graduate | Data Analyst |

# Future Enhancements

* Support for neutral sentiment

* Deep learning models (LSTM / BERT)

* Deployment on cloud (Render / AWS / Heroku)

* Improved UI with Bootstrap
