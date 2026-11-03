from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.shortcuts import render

from .forms import PredictionForm


MODEL_PATH = Path(__file__).resolve().parent / 'heart_disease_model.joblib'
FEATURE_NAMES = (
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
)


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


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
