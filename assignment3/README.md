# Assignment 3 code
This assignment was divided into 2 tasks:
- Implementing neural network code
- Creating classification CNN models.

## Section A
The code for this section is in `ch11.ipynb` file.
This file contains implementation of 2 layers linear model, first with numpy,
and second ith Tensorflow Keras.

The models trained on the MNIST datasets.

## Section B
The rest of the code files are for this section. It contains training code for
classification of 102 flowers dataset, based on `YOLOv5` and `VGG19` models.

The `YOLO` based model achieved an accuracy of 98.6% on the test set,
while the `VGG` model reached 90.3% accuracy. 
Both models trained for 20 epochs.

The weights for the `YOLO` model are available on this repo in the [weights](yolo-run-results/weights) folder,
and the VGG weights are available on [Google Drive](https://drive.google.com/drive/folders/1z3YZmTb_KOfdh0_uqoMeXZrIiyh2_kc-?usp=sharing).