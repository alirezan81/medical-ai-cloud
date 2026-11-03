from django import forms


class PredictionForm(forms.Form):
    age = forms.IntegerField(
        label='سن', min_value=18, max_value=120,
        widget=forms.NumberInput(attrs={'min': 18, 'max': 120, 'placeholder': 'مثلاً ۵۲'}),
    )
    sex = forms.TypedChoiceField(
        label='جنسیت ثبت‌شده', choices=[(0, 'زن (کد ۰)'), (1, 'مرد (کد ۱)')], coerce=int,
        widget=forms.Select(),
    )
    cp = forms.TypedChoiceField(
        label='نوع درد قفسه سینه', choices=[(i, f'کد {i}') for i in range(4)], coerce=int,
    )
    trestbps = forms.IntegerField(
        label='فشار خون در حالت استراحت (mmHg)', min_value=50, max_value=300,
        widget=forms.NumberInput(attrs={'min': 50, 'max': 300, 'placeholder': 'مثلاً ۱۲۵'}),
    )
    chol = forms.IntegerField(
        label='کلسترول سرم (mg/dl)', min_value=50, max_value=1000,
        widget=forms.NumberInput(attrs={'min': 50, 'max': 1000, 'placeholder': 'مثلاً ۲۱۲'}),
    )
    fbs = forms.TypedChoiceField(
        label='قند خون ناشتا بیشتر از ۱۲۰ mg/dl',
        choices=[(0, 'خیر (کد ۰)'), (1, 'بله (کد ۱)')], coerce=int,
    )
    restecg = forms.TypedChoiceField(
        label='نتیجه نوار قلب در حالت استراحت',
        choices=[(i, f'کد {i}') for i in range(3)], coerce=int,
    )
    thalach = forms.IntegerField(
        label='حداکثر ضربان قلب', min_value=30, max_value=250,
        widget=forms.NumberInput(attrs={'min': 30, 'max': 250, 'placeholder': 'مثلاً ۱۶۸'}),
    )
    exang = forms.TypedChoiceField(
        label='آنژین ناشی از ورزش', choices=[(0, 'خیر (کد ۰)'), (1, 'بله (کد ۱)')], coerce=int,
    )
    oldpeak = forms.FloatField(
        label='مقدار ST depression (oldpeak)', min_value=-10, max_value=20,
        widget=forms.NumberInput(attrs={'min': -10, 'max': 20, 'step': '0.1', 'placeholder': 'مثلاً ۱٫۰'}),
    )
    slope = forms.TypedChoiceField(
        label='شیب قطعه ST در اوج ورزش', choices=[(i, f'کد {i}') for i in range(3)], coerce=int,
    )
    ca = forms.TypedChoiceField(
        label='تعداد رگ‌های اصلی مشاهده‌شده', choices=[(i, str(i)) for i in range(5)], coerce=int,
    )
    thal = forms.TypedChoiceField(
        label='کد thal', choices=[(i, f'کد {i}') for i in range(4)], coerce=int,
    )
