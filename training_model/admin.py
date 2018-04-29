from django.contrib import admin

# Register your models here.

from training_model import models
admin.site.register(models.Data)