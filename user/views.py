from django.shortcuts import render
from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import IsAuthenticated,AllowAny
from django.contrib.auth.models import User
from rest_framework.response import Response

# Create your views here.

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    data = request.data
    firstName = data['firstName']
    lastName = data['lastName']
    password = data['password']
    username = data['email']
    try:
        User.objects.get(username=username)
        return Response(status=400)
    except:
        pass
    try:
        User.objects.create(username=username,password=password,first_name=firstName,last_name=lastName)
        return Response(status=200)
    except Exception as e:
        print(e)
        return Response(status=500)
    
    
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def getCurrUser(request):
    user = request.user
    context = {}
    context['email'] = user.username
    context['firstName'] = user.first_name
    context['lastName'] = user.last_name
    return Response(context)
    