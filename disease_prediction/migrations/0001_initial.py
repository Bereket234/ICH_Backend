# Generated by Django 4.1.7 on 2023-05-31 19:23

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('patient', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Disease',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('description', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Prediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('originalImage', models.ImageField(upload_to='')),
                ('predictedImage', models.ImageField(upload_to='')),
                ('hasDisease', models.BooleanField()),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('isBookmarked', models.BooleanField()),
                ('diseaseTypes', models.ManyToManyField(to='disease_prediction.disease')),
                ('patient', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='patient.patient')),
            ],
            options={
                'ordering': ['-date'],
            },
        ),
    ]
