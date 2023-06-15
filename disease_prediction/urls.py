from django.urls import path
from . import views


urlpatterns = [
    path("image/", views.predictImage),
    path("dicom/", views.predictDicom),
    path("10-days-data/", views.getTenDaysData),
    path("all-time-data/", views.getAllTimeData),
    path("all-predictions/", views.getAllPredictions),
    path("patient/<str:pk>/", views.getPredictionsByPatient),
    path("prediction/<str:pk>/", views.getPredictionById),
    path("get-zip-data/", views.getZippedData),
    path("get-bookmarks/", views.getBookmarkedPredictions),
    path("bookmark/<str:pk>/", views.bookmark),
]
