from django.shortcuts import render
from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import *
from .patientSerializers import *

# Create your views here.

#register a new patient 
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def registerPatient(request):
    data = request.data
    name = data['name']
    cardNo = data['cardNumber']
    age = data['age']
    description = data.get('description',None)
    sex = data['sex']
    phone = data['phone']
    try:
        Patient.objects.create(name=name,cardNumber=cardNo,age=age,description=description,sex=sex,phone=phone,imageCount=0)
        return Response(status=200)
    except Exception as e:
        print(e)
        return Response(status=500)
    
    
    
#get patient , delete patient or modify patient with id pk
@api_view(['DELETE','PUT','GET'])
@permission_classes([IsAuthenticated])
def patientApi(request,pk):
    if request.method == 'DELETE':
        try:
            patient = Patient.objects.get(id=pk)
        except Exception as e:
            print(e)
            return Response(status=404)
        patient.delete()
        return Response(status=200)
    
    
    if request.method == 'PUT':
        try:
            patient = Patient.objects.get(id=pk)
        except Exception as e:
            print(e)
            return Response(status=404)
        data = request.data
        patient.name = data['name']
        patient.cardNumber = data['cardNumber']
        patient.age = data['age']
        patient.sex = data['sex']
        patient.phone = data['phone']
        patient.description = data['description']
        try:
            patient.save()
            return Response(status=200)
        except Exception as e:
            return Response(status=500)
        
        
    if request.method == 'GET':
        try:
            patient = Patient.objects.get(id=pk)
        except Exception as e:
            print(e)
            return Response(status=404)
        serializer = PatientSerializer(patient)
        return Response(serializer.data)
    
    return Response(status=405)
    
    
    
#get all patients list
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def getPatients(request):
    try:
        patients = Patient.objects.all()
        serialiazer = PatientSerializer(patients,many=True)
        return Response(serialiazer.data)
    except Exception as e:
        print(e)
        return Response(status=500)


    