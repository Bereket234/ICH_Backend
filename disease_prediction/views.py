import io
import os
from wsgiref.types import FileWrapper
import zipfile
from django.http import FileResponse, HttpResponse
import tensorflow as tf
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
import cv2 as cv
import numpy as np
import pydicom
import urllib3
from django.core.files.base import ContentFile
from .models import *
from .predictionSerializers import *
from datetime import datetime, timedelta
from django.db.models import Q
from pandas import DataFrame as df

# Create your views here.

segmentationModel = tf.keras.models.load_model(
    "models/segmentation_model.hdf5", compile=False
)
classificationModel = tf.keras.models.load_model(
    "models/classification_model.h5", compile=False
)


# create a prediction
@api_view(["POST"])
@parser_classes([MultiPartParser])
@permission_classes([IsAuthenticated])
def predictImage(request):
    data = request.data
    requestImage = data["image"]
    patientId = data["patient"]
    try:
        patient = Patient.objects.get(id=patientId)
    except Exception as e:
        print(e)
        return Response(status=404)
    image = grab_image(stream=request.FILES["image"])
    predictedImage, mask, hasDisease, probabilities = segmentImage(image=image)
    intraventricular, intraparenchymal, subarachnoid, epidural, subdural = probabilities
    prediction = Prediction(
        patient=patient,
        originalImage=requestImage,
        isBookmarked=False,
        hasDisease=hasDisease,
        intraventricular=intraventricular,
        intraparenchymal=intraparenchymal,
        subarachnoid=subarachnoid,
        epidural=epidural,
        subdural=subdural,
    )
    prediction.save()

    ret, buf = cv.imencode(".jpg", predictedImage)
    content = ContentFile(buf.tobytes())
    prediction.predictedImage.save(str(prediction.id) + "_prediction.jpg", content)

    ret, buf = cv.imencode(".jpg", mask)
    content = ContentFile(buf.tobytes())
    prediction.mask.save(str(prediction.id) + "_mask.jpg", content)

    patient.imageCount += 1
    patient.save()

    serializer = PredictionSerializer(prediction)

    return Response(serializer.data)


# predict dicom images
@api_view(["Post"])
@permission_classes([IsAuthenticated])
def predictDicom(request):
    data = request.data
    dicom_data = request.FILES.get("image")
    if not dicom_data:
        return Response(status=400)
    ds = pydicom.dcmread(dicom_data)
    gray = bsb_window(ds)

    image = cv.resize(gray, (512, 512))
    image *= 255 / image.max()
    image = np.asarray(image, dtype=np.uint8)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    patientId = data["patient"]
    try:
        patient = Patient.objects.get(id=patientId)
    except Exception as e:
        print(e)
        return Response(status=404)
    predictedImage, mask, hasDisease, probabilities = segmentImage(image=image)
    intraventricular, intraparenchymal, subarachnoid, epidural, subdural = probabilities
    prediction = Prediction(
        patient=patient,
        isBookmarked=False,
        hasDisease=hasDisease,
        intraventricular=round(intraventricular * 100, 2),
        intraparenchymal=round(intraparenchymal * 100, 2),
        subarachnoid=round(subarachnoid * 100, 2),
        epidural=round(epidural * 100, 2),
        subdural=round(subdural * 100, 2),
    )

    prediction.save()

    ret, buf = cv.imencode(".jpg", image)
    content = ContentFile(buf.tobytes())
    prediction.originalImage.save(str(prediction.id) + "_image.jpg", content)

    ret, buf = cv.imencode(".jpg", predictedImage)
    content = ContentFile(buf.tobytes())
    prediction.predictedImage.save(str(prediction.id) + "_prediction.jpg", content)

    ret, buf = cv.imencode(".jpg", mask)
    content = ContentFile(buf.tobytes())
    prediction.mask.save(str(prediction.id) + "_mask.jpg", content)

    patient.imageCount += 1
    patient.save()

    serializer = PredictionSerializer(prediction)

    return Response(serializer.data)


# toggle bookmark of a prediction
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def bookmark(request, pk):
    try:
        pred = Prediction.objects.get(id=pk)
        pred.isBookmarked = not pred.isBookmarked
        pred.save()
        return Response(status=200)
    except:
        return Response(status=404)


# last 10 days data with hommerhage and without hommerhage
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getTenDaysData(request):
    days = []
    for i in range(10):
        currData = {}
        day = (datetime.now() - timedelta(days=i)).date()
        diseasedCount = Prediction.objects.filter(
            Q(date__startswith=day) & Q(hasDisease=True)
        ).count()
        totalCount = Prediction.objects.filter(date__startswith=day).count()
        dayString = day.strftime("%d-%m-%Y")
        currData["date"] = dayString
        currData["hemorrhage"] = diseasedCount
        currData["nohemorrhage"] = totalCount - diseasedCount
        days.append(currData)

    return Response(days)


# all time data hommerhage and no hommerhage counts
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getAllTimeData(request):
    context = {}
    diseasedCount = Prediction.objects.filter(hasDisease=True).count()
    freeCount = Prediction.objects.filter(hasDisease=False).count()
    context["hemorrhage"] = diseasedCount
    context["nohemorrhage"] = freeCount

    return Response(context)


# all predictions
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getAllPredictions(request):
    predictions = Prediction.objects.all()
    serializer = PredictionSerializer(predictions, many=True)
    return Response(serializer.data)


# get predicitions by patinet
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getPredictionsByPatient(request, pk):
    try:
        patient = Patient.objects.get(id=pk)
    except Exception as e:
        return Response(status=404)
    predictions = Prediction.objects.filter(patient=patient)
    serializer = PredictionSerializer(predictions, many=True)
    return Response(serializer.data)


# get prediction by id
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getPredictionById(request, pk):
    try:
        prediction = Prediction.objects.get(id=pk)
        serializer = PredictionSerializer(prediction)
        return Response(serializer.data)
    except Exception as e:
        print(e)
        return Response(status=404)


# download all time data zip
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getZippedData(request):
    zip_folder("mediafiles", "zipdata.zip")
    predictions = []
    predictionObjects = Prediction.objects.all()
    for p in predictionObjects:
        predictions.append(p.__dict__)
    predictions = df(predictions)
    if len(predictions) != 0:
        predictions = predictions.drop("_state", axis=1)
        predictions = predictions.drop("id", axis=1)
    predictions.to_csv("data.csv", index=False)
    with zipfile.ZipFile("zipdata.zip", mode="a") as zip_file:
        zip_file.write("data.csv")

    with open("zipdata.zip", "rb") as myfile:
        response = HttpResponse(
            io.BytesIO(myfile.read()), content_type="application/zip"
        )
        response["Content-Disposition"] = "attachment; filename=your_zipfile.zip"
    return response


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getBookmarkedPredictions(request):
    predictions = Prediction.objects.filter(isBookmarked=True)
    serilizer = PredictionSerializer(predictions, many=True)
    return Response(serilizer.data)


def segmentImage(image):
    image = cv.resize(image, (512, 512))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, th1 = cv.threshold(image, 250, 255, cv.THRESH_BINARY)
    th1 = cv.dilate(th1, kernel=np.ones((3, 3)), iterations=1)
    th1 = th1 == 0
    removedImage = image * th1
    removedImage = cv.cvtColor(removedImage, cv.COLOR_GRAY2BGR)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    segmentation = segmentationModel.predict(np.array([removedImage]))
    segmentation = segmentation[0]
    segmentation = np.amax(segmentation, axis=-1)
    segmentation[segmentation >= 0.5] = 1
    segmentation[segmentation < 0.5] = 0
    segmentation = segmentation * 255
    segmentation = cv.dilate(segmentation, kernel=np.ones((3, 3)), iterations=1)
    segmentation = np.array(segmentation, np.uint8)
    contours, hierarchy = cv.findContours(
        segmentation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1
    )
    padding = 20
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        paddedLeft = max(x - padding, 0)
        paddedTop = max(y - padding, 0)
        paddedRight = min(511, x + w + padding)
        paddedBottom = min(511, y + h + padding)
        cv.rectangle(
            image, (paddedLeft, paddedTop), (paddedRight, paddedBottom), (0, 0, 255), 5
        )
    if len(contours) == 0:
        probabilities = [0, 0, 0, 0, 0]
    else:
        probabilities = classificationModel.predict(np.array([removedImage]))[0]
    return image, segmentation, len(contours) > 0, probabilities


def grab_image(path=None, stream=None, url=None):
    if path is not None:
        image = cv.imread(path)
    else:
        if url is not None:
            resp = urllib3.request.urlopen(url)
            data = resp.read()
        elif stream is not None:
            data = stream.read()
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image


def correct_dcm(img):
    x = img.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    img.PixelData = x.tobytes()
    img.RescaleIntercept = -1000


def _window_image(img, window_center, window_width):
    if (
        (img.BitsStored == 12)
        and (img.PixelRepresentation == 0)
        and (int(img.RescaleIntercept) > -100)
    ):
        correct_dcm(img)

    img = img.pixel_array * img.RescaleSlope + img.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    # img = normalize(img)
    return img


def bsb_window(img):
    brain_img = _window_image(img, 40, 80)
    subdural_img = _window_image(img, 80, 200)
    soft_img = _window_image(img, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380

    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return brain_img


def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file))
