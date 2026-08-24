from django.urls import path
from . import views

urlpatterns = [

path('',views.home,name="home"),
path('image/',views.image_detection,name="image"),
path('webcam/',views.webcam_detection,name="webcam"),

]