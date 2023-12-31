# Generated by Django 4.2.1 on 2023-06-11 19:38

import disease_prediction.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('disease_prediction', '0004_remove_prediction_diseasetypes_delete_disease'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prediction',
            name='mask',
            field=models.ImageField(blank=True, null=True, upload_to=disease_prediction.models.upload_to_mask),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='originalImage',
            field=models.ImageField(blank=True, null=True, upload_to=disease_prediction.models.upload_to_image),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='predictedImage',
            field=models.ImageField(blank=True, null=True, upload_to=disease_prediction.models.upload_to_prediction),
        ),
    ]
