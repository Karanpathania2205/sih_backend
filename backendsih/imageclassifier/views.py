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
from scipy import stats
import scipy
import glob 
import cv2
import pandas as pd 
import numpy as np
from skimage import feature
from skimage import filters
from skimage import data
import sklearn
from sklearn.cluster import KMeans
#atrribute selection imports
import torch 
import torch.nn as nn
from torchvision import models
from torchsummary import summary 
import torchvision.transforms as transforms
import glob 
from sklearn.cluster import KMeans
import cv2
import numpy as np
import pandas as pd 
#segmentation imports
# import pixellib
# from pixellib.torchbackend.instance import instanceSegmentation
# from pixellib.semantic import semantic_segmentation
# import numpy as np
# import cv2
# import pytesseract as tr


def home(request):
    return render(request,'home.html')


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
    centers=None
    features=None
    if request.method == 'POST':
        print (request)
        print (request.POST.dict())
        fileObj=request.FILES['filePath']
        fs=FileSystemStorage()
        filePathName=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(filePathName)
        print(filePathName)
        image="."+filePathName
        print(image)

        #segmentation 

        #feature extraction
        path = "/content/drive/MyDrive/SIH/Images2/"
        #$weights =  models.ResNet50_Weights.DEFAULT
        model = models.vgg16(pretrained = True)

        model = model.type(torch.cuda.FloatTensor)
        print(summary(model,(3,256,256)))

        class extract(nn.Module):
            def __init__(self,model):
                super(extract,self).__init__()
                self.features  = list(model.features)
                self.features = nn.Sequential(*self.features)
                self.pooling = model.avgpool
                self.flatten = nn.Flatten()
                #self.linear = nn.Linear(1,10)
                self.fc = model.classifier[1]

            def forward(self,input):
                out = self.features(input)
                out = self.pooling(out)
                out = self.flatten(out)
                out = self.fc(out)
                return out 
        #model = models.resnet50(weights = weights)

        updated = extract(model)
        print(summary(updated,(3,256,256)))

        transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
        features = []
        result  = glob.glob(path+"/*.jpg")
        for i,path1 in enumerate(result):

            image = cv2.imread(path1)
            image = transform(image)
            image = image.type(torch.cuda.FloatTensor)
        #image = image.to(device)
            #print(type(image))
        #feature = updated(image)
            with torch.no_grad():
                feature = updated(image)
            features.append(feature.cpu().detach().numpy().reshape(-1))
        features = np.array(features)
        with open("/content/drive/MyDrive/SIH/features"+str(i)+".txt",'a') as f:

            f.write(str(features))
            f.write("\n")

        print(features)

        #features = features.reshape(1,-1)
        clust = KMeans(n_clusters = 1,random_state= 0)
        #features  =features.cpu()
        clust1 = clust.fit(features)
        labels = clust1.labels_
        number = clust1.n_features_in_
        name = clust1.feature_names_in_
        print(clust1)
        print(number)
        print(name)
        #labels.to_csv("labels.csv",index = False)
        #np.savetxt("/content/drive/MyDrive/SIH/labels.txt",labels)


    context = {
        'filePathName':filePathName,
        'centers':centers,
        'features':features,
    }
    return render(request, 'index.html', context)