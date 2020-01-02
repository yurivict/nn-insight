# NN Insight

NN Insight a program that visualizes neural network's structure and computation results.

## Features
* Visually shows all obects and data that neural network contains: operators, tensors, static arrays, operator options.
* Allows to run neural networks on images and see results immediately.
* Shows all intermediate layers, both as tables with numbers and as black-and-white images.
* Performs analytics on neural networks, shows how relatively expensive different operators are, and how much data they generate or operate on.

## Supported NN formats
* TF Lite (only floating point models, not quantized models)

## Supported networks
* MobileNet V1 (download the file: https://drive.google.com/file/d/1FYK72GkbqJUwgFZ8q_32HtI7X3CrfBtT/view?usp=sharing)
* MobileNet V2 (download the file: https://drive.google.com/file/d/1XicUqcqUNa14DfqeWyUYHZGHDXO5XAgp/view?usp=sharing)
* MobileNet V3 (download the file: https://drive.google.com/file/d/1qq6xLx98M_wy9YqetOxuqNJCSOEq5O8f/view?usp=sharing)

MobileNet networks are a series of networks that Google released over the past few years that are trained on the ImageNet image set with 1001 categories.
Other networks might also work, but only three above networks have been verified.

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

