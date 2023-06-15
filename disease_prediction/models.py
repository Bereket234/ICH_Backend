from django.db import models

from patient.models import Patient

# Create your models here.


def upload_to_image(instance, filename):
    return "images/{filename}".format(filename=filename)


def upload_to_prediction(instance, filename):
    return "predictions/{filename}".format(filename=filename)


def upload_to_mask(instance, filename):
    return "masks/{filename}".format(filename=filename)


class Prediction(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True)
    originalImage = models.ImageField(null=True, blank=True, upload_to=upload_to_image)
    predictedImage = models.ImageField(
        null=True, blank=True, upload_to=upload_to_prediction
    )
    mask = models.ImageField(null=True, blank=True, upload_to=upload_to_mask)
    hasDisease = models.BooleanField()
    date = models.DateTimeField(auto_now_add=True)
    isBookmarked = models.BooleanField()
    intraventricular = models.FloatField()
    intraparenchymal = models.FloatField()
    subarachnoid = models.FloatField()
    epidural = models.FloatField()
    subdural = models.FloatField()

    def __str__(self) -> str:
        return self.patient.name[:10] + str(self.hasDisease)

    class Meta:
        ordering = ["-date"]
