from django.urls import path
from . import views

urlpatterns = [
    path('register/',views.registerPatient),
    path('all/',views.getPatients),
    path('<str:pk>/',views.patientApi),
    
]
