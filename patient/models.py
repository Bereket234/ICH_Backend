from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Patient(models.Model):
    name = models.CharField(max_length=50)
    cardNumber = models.IntegerField()
    age = models.IntegerField()
    sex = models.CharField(max_length=10)
    phone = models.CharField(max_length=20)
    description = models.TextField()
    registeredDate = models.DateTimeField(auto_now_add=True)
    imageCount = models.IntegerField()
    
    
    def __str__(self):
        return self.name[:20]
    
    class Meta:
        ordering = ['-registeredDate']