from django.shortcuts import render
from django.conf.urls import url
from . import views
# Create your views here.
app_name = 'NewsCompanion'
urlpatterns = [

url(r'home', views.home, name='home'),
url(r'result', views.result, name='result'),
url(r'pie', views.pie, name='pie'),
url(r'about', views.about, name='about'),



]
