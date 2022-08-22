from django.shortcuts import render,HttpResponse,redirect
from django.contrib import messages
from .forms import ImageUploadForm
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User, auth
import torch
from torchvision import transforms
import numpy as np
import glob 
import cv2
import base64

def register(request):
    if request.method=='POST':
        first_name=request.POST['first_name']
        last_name=request.POST['last_name']
        username=request.POST['username']
        password1=request.POST['password1']
        password2=request.POST['password2']
        email=request.POST['email']
        if password1==password2:
            if User.objects.filter(username=username).exists():
                messages.info(request,"Username taken")
            elif User.objects.filter(email=email).exists():
                messages.info(request,"Email exists")
            else:
                user=User.objects.create_user(username=username,password=password1,email=email,first_name=first_name,last_name=last_name)
                user.save();
                messages.info(request,'user created')
                return redirect('login')
        else:
            messages.info(request,"password not matching")

        return redirect('/')
    else:
        return render(request,'register.html')

def login(request):
    if request.method=='POST':
        username=request.POST['username']
        password=request.POST['password']

        user=auth.authenticate(username=username,password=password)

        if user is not None:
            auth.login(request,user)
            return redirect("image")
        else:
            messages.info(request,'invalid credentials')
            return redirect('login')
    else:
        return render(request,'login.html')

def logout(request):
    auth.logout(request)
    return redirect('/')

def index(request):
    filePathName=None
    if request.method == 'POST':
        print (request)
        print (request.POST.dict())
        fileObj=request.FILES['filePath']
        fs=FileSystemStorage()
        filePathName=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(filePathName)
        image='.'+filePathName
    context = {
        'filePathName':filePathName,
        # 'centers':centers,
    }
    return render(request, 'index.html', context)