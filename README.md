# Adaptively-adjusted-threshold-for-SCNN
This project is the source code for thesis ” An adaptive threshold mechanism for accurate and efficient deep spiking convolutional neural networks“

1、This code is based on the SNN-TB platform (https://github.com/NeuromorphicProcessorProject/snn_toolbox.git) to convert the ANN into SNN and simulate it.

2、We modify the source code of SNN-TB platform to implement our algorithm

3、The source code of the algorithm proposed in our paper which adaptively adjusts SNN threshold value is in 
  \snntoolbox\simulation\backends\inisim\temporal_mean_rate_tensorflow.py
  
4、How to reproduce our experimental data：
  a）As for the cifar10 dataset：
   You must first train the ANN model of VGG19('Vgg19_keras_cifar10.py'), which will generate a h5 file to store the VGG19 trained model and weight data.Then, run the file 'SNN_Simulation_cifar10.py', which will convert the trained VGG19 into an SNN and simulate it according to our algorithm.
   
  b）As for the cifar100 dataset：
   You must first train the ANN model of VGG19('Vgg19_keras_cifar100.py'), which will generate a h5 file to store the VGG19 trained model and weight data.Then, run the file 'SNN_Simulation_cifar100.py', which will convert the trained VGG19 into an SNN and simulate it according to our algorithm.
   
Note: you must make sure that you have snntoolbox folder in your project file, and after simulating SNN on cifar10, you need to delete the data generated by the implementation, so as to prevent errors in the experiment on cifar100 dataset
