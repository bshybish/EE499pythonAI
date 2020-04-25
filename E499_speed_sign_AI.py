# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:50:17 2020

@author: Bassam
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image as img
from os import listdir
from os.path import isfile, join

mypath = "C:\\Users\\Bassam\\Documents\\training_data\\first_set\\"


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
filooos = listdir(mypath)
s_filooos = sorted(filooos)


