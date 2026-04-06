# Hate Speech Detector

This is now a Django + machine learning application that trains and predicts locally without any API.

The web app lets you:
- upload a small CSV dataset
- train a TF-IDF + Logistic Regression model
- predict hate speech from a textbox
- keep everything on your machine

## Project Structure

```text
hate_speech/
├── detector/
│   ├── forms.py
│   ├── urls.py
│   ├── views.py
│   └── templates/
├── hate_speech_project/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── src/
│   ├── agent.py
│   ├── config.py
│   ├── data.py
│   ├── text_cleaning.py
│   └── utils.py
├── data/
├── models/
├── reports/
├── static/
├── manage.py
├── requirements.txt
└── README.md
```

## Dataset Format

Upload a CSV file with these columns:

```csv
text,label
I hate you,1
Have a nice day,0
```

Accepted labels:
- `1`, `hate`, `hateful`, `offensive`, `toxic`, `abusive`
- `0`, `not_hate`, `normal`, `clean`, `neutral`, `safe`

## Setup

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run migrations with `python manage.py migrate`.
4. Start the server with `python manage.py runserver`.
5. Open `http://127.0.0.1:8000/`.
6. Upload your dataset and train the model.
7. Type text into the prediction box and submit.

## How It Works

The Django view saves your uploaded CSV to `data/raw/hate_speech.csv`, then calls the local training code in `src/agent.py`. The trained model is saved in `models/hate_speech_model.joblib`, and predictions are produced from the same local model.

## Notes

- No external API is used.
- This is a good beginner-friendly baseline project.
- If you want, the next step can be adding a database table for saved prediction history or separating training into a background task.
