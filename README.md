# Sound To Tactile

This code is part of assignment for the course - Introduction to haptics (EEL7670) as part of MTech - AR/VR Programme by IIT Jodhpur under the guidance of Dr. Amit Bhardwaj.

This code implements the paper - [Method for Audio-to-Tactile Cross-Modality Generation Based on Residual U-Net](https://ieeexplore.ieee.org/document/10330031)

The ST-SIM measure is taken from the paper - [Vibrotactile Signal Compression Based on Sparse Linear Prediction and Human Tactile Sensitivity Function](https://ieeexplore.ieee.org/document/8816110)

## Problem statement

Wielding a stylus to explore the surface or texture of an object generates signals in two modalities Sound and Haptics. There are ample literature that shows that there is mapping between these two modalities. Here in this assignment we ask you (A)to find such a mapping and (B)evaluate its performance based on some measure. Formally,
1. Develop an approach to create models that realize automatic cross modal signal generation, pertaining to certain exploratory procedures.
2. Use ST-SIM measure defined in the paper(Vibrotactile Signal Compression Based on Sparse Linear Prediction and Human Tactile Sensitivity Function.) to gauge performance of your algorithm.

## Setting up the project

To setup the project you just need to install and setup anaconda in your computer. This project is created in python-3.11.8.

1. Setup the environment
    1. Use the following command in the root of the project to setup the conda environment.

    ```
    conda env create -f conda-environment.yml
    ```

    2. Activate the newly created conda environment.

    ```
    conda activate deep_learning
    ```

2. Setting up the directories

    1. Create a directory called output. Create the following directories inside output directory.
        1. model
        2. preprocessing
    2. Create a directory called data. Download the LMT Haptic database(108 surface materials)  - https://zeus.lmt.ei.tum.de/downloads/texture/#oldtextures . Extract the dataset into this data directory. So, it should look like data -> AccelScansComponent, data -> SoundScans

## Running the code

1. Make sure to activate the conda environment before trying to run the code.
2. For training you can run `python main.py`
3. For evaluating the model you can run `python eval.py`
4. For getting the ST-SIM measure you can run `python similarity_eval.py`

## Downloading pretrained weights

you can download pretrained weights from here -
https://drive.google.com/file/d/1DLv4xiQaU3X4j_cPMhSgoFRfXAVarBPD/view?usp=drive_link

Download the weights and put them inside the `output -> model` directory.

## References 

1. R. Hassen and E. Steinbach, "Vibrotactile Signal Compression Based on Sparse Linear Prediction and Human Tactile Sensitivity Function," 2019 IEEE World Haptics Conference (WHC), Tokyo, Japan, 2019, pp. 301-306, doi: 10.1109/WHC.2019.8816110. keywords: {Sensitivity;Acceleration;Encoding;Vibrations;Predictive models;Minimization;Surface waves}

2. Y. Zhan, X. Sun, Q. Wang and W. Nai, "Method for Audio-to-Tactile Cross-Modality Generation Based on Residual U-Net," in IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-14, 2024, Art no. 2500414, doi: 10.1109/TIM.2023.3336453.
