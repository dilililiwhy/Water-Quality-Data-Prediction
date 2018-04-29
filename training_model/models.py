from django.db import models

# Create your models here.


class Data(models.Model):
    station = models.CharField(max_length=30)
    time = models.IntegerField()
    pH = models.FloatField(null=True)
    TN = models.FloatField(null=True)
    TP = models.FloatField(null=True)
    NH4 = models.FloatField(null=True)
    COD = models.FloatField(null=True)
    DO = models.FloatField(null=True)

class p_Data(models.Model):
    station = models.CharField(max_length=30)
    time = models.IntegerField()
    NH4 = models.FloatField(null=True)
    COD = models.FloatField(null=True)
