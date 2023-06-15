from rest_framework.serializers import ModelSerializer, SerializerMethodField
from .models import *


class PredictionSerializer(ModelSerializer):
    patient = SerializerMethodField()
    date = SerializerMethodField()

    def get_patient(self, object):
        currPatient = object.patient
        return {
            "id": currPatient.id,
            "name": currPatient.name,
            "cardNumber": currPatient.cardNumber,
            "age": currPatient.age,
            "sex": currPatient.sex,
            "phone": currPatient.phone,
            "description": currPatient.description,
            "registeredDate": currPatient.registeredDate.date().strftime("%d-%m-%Y"),
            "imageCount": currPatient.imageCount,
        }

    def get_date(self, object):
        return object.date.date().strftime("%d-%m-%Y")

    class Meta:
        model = Prediction
        fields = [
            "id",
            "patient",
            "originalImage",
            "predictedImage",
            "hasDisease",
            "date",
            "isBookmarked",
            "mask",
            "intraventricular",
            "intraparenchymal",
            "subarachnoid",
            "epidural",
            "subdural",
        ]
