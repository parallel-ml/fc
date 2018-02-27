# Siamese_clsfy

#### keras_quick_start
This file stores all the code for 4 different units of machine learning model, 
including temporal, spatial, maxpool and fc. 

#### util/output
This file is useful for having good format and timing the model inference time. It uses
a decorator to wrap a function and examine the run time. Check ```siamese_fc_layer.py```
for how to use it. 

#### units in ML model
* spatial and temporal: They are both based on CNN (convolution neural network). The only
difference between them are input shapes. The spatial is for single image, but the temporal
is for optical flow, which is a combination of multiple frames.
* maxpooling: Simple unit is used to extract maximum value from a certain size of input.
The maxpooling unit has 4 maxpooling layers, but each with different stride size.
* fc: Fully connected layer.

Please refer to ```keras_quick_start.py``` to see examples of running different units. The 
fc layer is under discussion, so take a look at spatial, temporal and maxpool first. 
 