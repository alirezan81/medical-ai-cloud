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


def _binary_choices():
    return [(0.0, 'خیر (کد ۰)'), (1.0, 'بله (کد ۱)')]


class DiabetesPredictionForm(forms.Form):
    HighBP = forms.TypedChoiceField(label='فشار خون بالا', choices=_binary_choices(), coerce=float)
    HighChol = forms.TypedChoiceField(label='کلسترول بالا', choices=_binary_choices(), coerce=float)
    CholCheck = forms.TypedChoiceField(label='آزمایش کلسترول در ۵ سال گذشته', choices=_binary_choices(), coerce=float)
    BMI = forms.FloatField(
        label='شاخص توده بدنی (BMI)', min_value=12, max_value=98,
        widget=forms.NumberInput(attrs={'min': 12, 'max': 98, 'step': '0.1', 'placeholder': 'مثلاً ۲۷٫۵'}),
    )
    Smoker = forms.TypedChoiceField(label='سابقه مصرف حداقل ۱۰۰ سیگار', choices=_binary_choices(), coerce=float)
    Stroke = forms.TypedChoiceField(label='سابقه سکته مغزی', choices=_binary_choices(), coerce=float)
    HeartDiseaseorAttack = forms.TypedChoiceField(
        label='سابقه بیماری قلبی یا حمله قلبی', choices=_binary_choices(), coerce=float,
    )
    PhysActivity = forms.TypedChoiceField(label='فعالیت بدنی در ماه گذشته', choices=_binary_choices(), coerce=float)
    Fruits = forms.TypedChoiceField(label='مصرف میوه روزانه', choices=_binary_choices(), coerce=float)
    Veggies = forms.TypedChoiceField(label='مصرف سبزیجات روزانه', choices=_binary_choices(), coerce=float)
    HvyAlcoholConsump = forms.TypedChoiceField(
        label='مصرف زیاد الکل', choices=_binary_choices(), coerce=float,
    )
    AnyHealthcare = forms.TypedChoiceField(label='پوشش خدمات درمانی', choices=_binary_choices(), coerce=float)
    NoDocbcCost = forms.TypedChoiceField(
        label='مراجعه نکردن به پزشک به دلیل هزینه', choices=_binary_choices(), coerce=float,
    )
    GenHlth = forms.TypedChoiceField(
        label='ارزیابی کلی سلامت', choices=[(float(i), f'کد {i}') for i in range(1, 6)], coerce=float,
    )
    MentHlth = forms.IntegerField(
        label='روزهای با سلامت روان نامطلوب (در ۳۰ روز گذشته)', min_value=0, max_value=30,
        widget=forms.NumberInput(attrs={'min': 0, 'max': 30}),
    )
    PhysHlth = forms.IntegerField(
        label='روزهای با سلامت جسمی نامطلوب (در ۳۰ روز گذشته)', min_value=0, max_value=30,
        widget=forms.NumberInput(attrs={'min': 0, 'max': 30}),
    )
    DiffWalk = forms.TypedChoiceField(label='دشواری در راه رفتن یا بالا رفتن از پله', choices=_binary_choices(), coerce=float)
    Sex = forms.TypedChoiceField(
        label='جنسیت ثبت‌شده', choices=[(0.0, 'زن (کد ۰)'), (1.0, 'مرد (کد ۱)')], coerce=float,
    )
    Age = forms.TypedChoiceField(
        label='گروه سنی', choices=[(float(i), f'گروه سنی {i}') for i in range(1, 14)], coerce=float,
    )
    Education = forms.TypedChoiceField(
        label='سطح تحصیلات', choices=[(float(i), f'کد {i}') for i in range(1, 7)], coerce=float,
    )
    Income = forms.TypedChoiceField(
        label='گروه درآمد', choices=[(float(i), f'کد {i}') for i in range(1, 9)], coerce=float,
    )
