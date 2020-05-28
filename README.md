# NN Insight

NN Insight is a program that visualizes neural network's structure and computation results. NN Insight is currently a in its alpha stage, and is under development.

## Features
* Visually shows all objects and data that neural network contains: operators, tensors, static arrays, operator options.
* Allows to run neural networks on images and see results immediately.
* Shows all intermediate layers, both as tables with numbers and as black-and-white images.
* Performs analytics on neural networks, shows how relatively expensive different operators are, and how much data they generate or operate on.
* Allows to easily run on any image region selected visually.
* Allows to copy and paste images.
* Allows to apply effects to images before processing.
* Doesn't require knowledge of any programming languages.

## Supported NN formats
* TF Lite (only floating point models, not quantized models)

## Supported networks
* MobileNet V1 [[link](https://drive.google.com/file/d/1FYK72GkbqJUwgFZ8q_32HtI7X3CrfBtT/view?usp=sharing)]
* MobileNet V2 [[link](https://drive.google.com/file/d/1XicUqcqUNa14DfqeWyUYHZGHDXO5XAgp/view?usp=sharing)]
* MobileNet V3 [[link](https://drive.google.com/file/d/1qq6xLx98M_wy9YqetOxuqNJCSOEq5O8f/view?usp=sharing)]
* VGG16 [[link](https://drive.google.com/file/d/1Nw6a_PcoQi4ZaTEFLO6J8KJvP-ZpgXPi/view?usp=sharing)]
* VGG19 [[link](https://drive.google.com/file/d/1wsNsQRknfKUgS_zp6kLLETE9ngdd1pYA/view?usp=sharing)]
* SqueezeNet [[link](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz)]
* Fire-Detection-CNN [[link](https://drive.google.com/file/d/17wuKEUUL2ONkPMiYExh8Bx6M6lAUGbxA/view?usp=sharing)] from the paper _Dunnings and Breckon, In Proc. International Conference on Image Processing IEEE, 2018_ [[pdf](https://breckon.org/toby/publications/papers/dunnings18fire.pdf)] (use normalizaton 0..255 and BGR)

MobileNet networks are a series of networks that Google released over the past few years that are trained on the ImageNet image set with 1001 categories.

VGG16 and VGG19 networks are deep convolutional networks proposed by K. Simonyan and A. Zisserman from the University of Oxford Visual Geometry Group in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. [[link](https://arxiv.org/abs/1409.1556)]

## How to use
1. Download a TF Lite file using one of the links above, or use a file downloaded elsewhere.
2. Start NN Insight: 'nn-insight {file.tflite}'.
3. Paste some image (Ctrl-V).
4. Press "Compute".
5. See what the network thinks you have pasted.
6. Zoom the image using the 'Scale image' widget to focus on some other object, and see if network's answer would change.

## NN Insight is alpha software
The NN Insight project was only started on Dec 20th 2019, and it is in its early stages. It will see a lot of developments in the coming time.

## Limitations
* Many operators aren't supported yet.
* Quantized models aren't supported yet.
* Intermediate layer display isn't as sophisticated as it could be.
* 1D data display is missing.
* Scrolling issues are present in the neural network view.
* Not all UI events are yet properly connected.

## Screenshot
![Alt text](https://raw.githubusercontent.com/yurivict/nn-insight/master/screenshot.png "NN Insight")

