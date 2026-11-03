from django.urls import path

from . import views

app_name = 'prediction'

urlpatterns = [
    path('', views.home, name='home'),
]
