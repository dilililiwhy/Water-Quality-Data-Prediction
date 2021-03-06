# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-04-27 11:21
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('training_model', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data',
            name='BOD',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='data',
            name='COD',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='data',
            name='DO',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='data',
            name='NH4',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='data',
            name='TN',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='data',
            name='TP',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='data',
            name='chlorphylla',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='data',
            name='pH',
            field=models.FloatField(null=True),
        ),
    ]
