from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.shortcuts import render

from .forms import DiabetesPredictionForm, PredictionForm


MODEL_PATH = Path(__file__).resolve().parent / 'heart_disease_model_deduplicated.joblib'
FEATURE_NAMES = (
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
)
DIABETES_MODEL_PATH = Path(__file__).resolve().parent / 'diabetes_model.joblib'
DIABETES_FEATURE_NAMES = (
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
)


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_diabetes_model():
    return joblib.load(DIABETES_MODEL_PATH)


def home(request):
    result = None
    form = PredictionForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        values = {name: form.cleaned_data[name] for name in FEATURE_NAMES}
        input_data = pd.DataFrame([values], columns=FEATURE_NAMES)
        prediction = int(load_model().predict(input_data)[0])
        result = 'مدل، ورودی را در گروه دارای بیماری قلبی قرار داد.' if prediction == 1 else (
            'مدل، ورودی را در گروه بدون بیماری قلبی قرار داد.'
        )

    return render(request, 'prediction/home.html', {'form': form, 'result': result})


def diabetes(request):
    result = None
    form = DiabetesPredictionForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        values = {name: form.cleaned_data[name] for name in DIABETES_FEATURE_NAMES}
        input_data = pd.DataFrame([values], columns=DIABETES_FEATURE_NAMES)
        try:
            prediction = int(load_diabetes_model().predict(input_data)[0])
        except Exception:
            result = 'در پردازش اطلاعات خطایی رخ داد. لطفاً بعداً دوباره تلاش کنید.'
        else:
            result = 'مدل، ورودی را در گروه دیابت قرار داد.' if prediction == 1 else (
                'مدل، ورودی را در گروه بدون دیابت قرار داد.'
            )

    return render(request, 'prediction/diabetes.html', {'form': form, 'result': result})
