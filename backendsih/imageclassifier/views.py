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

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            print (request)
            print (request.POST.dict())
            fileObj=request.FILES['filePath']
            fs=FileSystemStorage()
            filePathName=fs.save(fileObj.name,fileObj)
            filePathName=fs.url(filePathName)
            filePathName='.'+filePathName
            image='.'+filePathName
            #scene infrence
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
            feat = []
            features = []
            path = image
            result = glob.glob(path+'/*.jpg')
            for j,path1 in enumerate(result):
                image = cv2.imread(path1)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                
                
                mean  = scipy.mean(image,axis=0)
                

                
                mean = mean.tolist()
                std = scipy.std(image,axis=0)
                std = std.tolist()
                

                skewness = stats.skew(image,axis=0)
                skewness = skewness.tolist()
                skewness = np.concatenate((mean,std,skewness),axis= 1)
                norm_mom = np.linalg.norm(skewness)
                features.append(norm_mom)





                hue = image[0]
                saturation  = image[1]
                meana = scipy.mean(hue,axis = 0)
                

                stda= scipy.std(hue,axis = 0)
                #print(std)
                

                mean1 = scipy.mean(saturation,axis = 0)
                #print(mean1)
                

                std1 = scipy.std(saturation,axis = 0)
                #print(std1)


                con = np.concatenate((meana,stda,mean1,std1),axis = 0)
                #print(con)
                norm_con = np.linalg.norm(con)
                features.append(norm_con)
                image  = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
                can = cv2.Canny(image,100,200)
                can = np.array(can)
                norm_can = np.linalg.norm(can)

                features.append(norm_can)

                gabor  = filters.frangi(image)
                image = image[:,:,:]
                #print(gabor.shape)
                df = pd.DataFrame()
                num = 1

                for i in range(1,6):
                    i = (i/4)*3.14
                    for h in range(3,9):
                        for j in range(1,4):
                            for k in range(1,2):

                                kernel = cv2.getGaborKernel((39,39),h,j,i,k,0,ktype=cv2.CV_32F)
                                img  = cv2.filter2D(image,cv2.CV_8UC3,kernel)
                                img = img.reshape(-1)
                            #df[Gabor] = img
                            #df.to_csv("Gabor.csv")
                                num = num + 1
                #print(img)
                gab = np.array(img)
                gab_norm = np.linalg.norm(gab)
                features.append(gab_norm)

                
                image = image[:,:,0]
                mat = feature.graycomatrix(image,[1],[45])
                tamura = feature.graycoprops(mat,prop = 'contrast')
                tamura1 = feature.graycoprops(mat,prop = 'dissimilarity')
                tamura2 = feature.graycoprops(mat,prop = 'homogeneity')
                #with open("chromatic.txt",'a') as g:
                    #g.write(str(con))
                    #g.write('\n')


                arr1= np.concatenate((tamura,tamura1,tamura2),axis = 1)
                #print(arr.shape)
                #arr= arr.reshape(arr[0],arr[1]*arr[2])

                #arr = arr.flat()
                #print(dir(arr))
                #arr = features.tolist()
                norm_tam = np.linalg.norm(arr1)
                #print(norm)


                
                #print(arr)
                
                #print(arr.size)
                
                #print(arr)
                features.append(norm_tam)
            #print(feat)
            features  = np.array(features)
            features = features.reshape(-1,1)
            clust = KMeans(n_clusters = 1,random_state =0)
            clust1 = clust.fit(feat)
            centers = clust1.cluster_centers_
            try:
                predicted_label ="something"
            except RuntimeError as re:
                print(re)
                # predicted_label = "Prediction Error"

    else:
        form = ImageUploadForm()

    context = {
        'filePathName':filePathName,
        'centers':centers,
    }
    return render(request, 'index.html', context)