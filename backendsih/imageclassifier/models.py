from django.db import models

# Create your models here.
class ResultImage(models.Model):
    image_id=models.AutoField
    image=models.ImageField(upload_to="images")