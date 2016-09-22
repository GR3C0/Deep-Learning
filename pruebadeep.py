#-*- coding: utf-8 -*-
#pruebas sobre el deep learning utilizando la libreria tensorflow de google

import tensorflow as tf
import input_data # archivo para leer la libreria de imagenes MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
	