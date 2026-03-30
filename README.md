<h1 align="center">ExoFILT</h1>
<h3 align="center">Transfer learning for robust and accelerated analysis of exocytosis single-particle tracking data</h3>

**ExoFILT** is a Deep Learning binary classifier designed to identify **bona fide exocytic events** from live-cell imaging data in *Saccharomyces cerevisiae*.

For more details about this method, see [ExoFILT: Transfer learning for robust and accelerated analysis of exocytosis single-particle tracking data](https://www.biorxiv.org/content/10.64898/2026.02.27.708581v1)

## Overview

This repository provides a complete pipeline for the analysis of exocytosis events from live-cell imaging data. It contains two main components:

1. **ImageJ/Fiji scripts**  
   Tools for preprocessing microscopy movies, performing automated tracking, and manually annotating tracks.
   
2. **Neural network inference**  
   A Jupyter notebook that applies **ExoFILT** to classify exocytic events.

Pretrained ExoFILT models used for inference are provided in the repository under the `models/` directory.

## Installation (Neural Network inference)

### 1. Clone the repository

```bash
git clone https://github.com/GallegoLab/ExoFILT.git
cd ExoFILT
```

### 2. Create and activate the Conda environment

Two environments are provided, one for easy installation (using only CPU) and one for fast Neural Network inference using GPU. Note that the GPU environment does not support the usage of the streamlit package due to different dependencies, so use the CPU environment for data visualization. 

#### Easy installation: CPU Environment

Create the conda environment and install the required packages:
```bash
conda env create -f environment_cpu.yml
```

After installation, activate the environment:
```bash
conda activate exofilt_cpu
```

#### Advanced installation: GPU Environment

If a GPU is available, follow these steps to increase the inference speed.
```bash
conda env create -f environment_gpu.yml
```

After installation, activate the environment:
```bash
conda activate exofilt_gpu
```

Set up environment variables for GPU:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d \
&& echo -e '#!/bin/sh\nexport LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh \
&& chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh \
&& echo -e '#!/bin/sh\nunset LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh \
&& chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

**Restart the environment** (to make the new variables take effect):
```bash
conda deactivate
conda activate exofilt_gpu
```

Check that TensorFlow detects the GPU:
```bash
python - <<END
import tensorflow as tf
print("\n\nTensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus, "\n\n")
END
```

If everything works, at least one GPU should appear listed.


## ImageJ / Fiji Preprocessing & Annotation

### Requirements
The ImageJ scripts were developed and tested with:
  * ImageJ 1.54p
  * Java 1.8.0_322
  * TrackMate v7.14.0
  * Jython scripting enabled
  * Accurate Gaussian Blur plugin. 
  
A standard **Fiji** installation should contain the required dependencies, except `Accurate Gaussian Blur` plugin (see below installation details).
  
### Scripts

1) `1_Preprocessing.py`: Preprocessing of simultaneous dual-color live-cell imaging movies.
2) `2_Tracking.py`: Automated particle tracking of preprocessed movies using TrackMate.
3) `3_Annotation_GUI.py`: GUI for manual annotation of individual tracks in channel 1.
4) `4_Colocalization_GUI.py`: GUI for manual annotation of colocalization between channels.

### Example data

All raw movies used in the ExoFILT study are available in a [Zenodo repository](https://zenodo.org/records/18962705). These datasets can be used to test the workflow. 

The repository also includes **CSV files containing track annotations**, which can be loaded with the `3_Annotation_GUI.py` script to visualize examples of **bona fide** and **ambiguous** exocytic events.

### Workflow

#### 1) Preprocessing
Place raw movies in a folder named:
```
1_Raw
```
Open `1_Preprocessing.py` in ImageJ and run it with Jython (*Language &rarr; Jython*). 

After selecting the appropiate parameters in the initial dialog, the script creates two new directories:
```
2_Splitted
3_Preprocessed
```
These folders contain channel-separated movies, before and after preprocessing, respectively.

#### 2) Automated Tracking
To perform automated tracking, the plugin **`Accurate Gaussian Blur`** is required. To install this plugin, download [Accurate_Gaussian_Blur.class](https://imagej.net/ij/plugins/download/Accurate_Gaussian_Blur.class), copy to the plugins folder and restart ImageJ. 

Open `2_Tracking.py` in ImageJ and run it with Jython. 
 
The script performs automated tracking using **TrackMate** on the C1 movies located in:
```
3_Preprocessed
```
Tracking results are stored as:
```
tracks_all.csv
```
All tracking outputs are written to a new folder:
```
4_Tracking
```

#### 3) ExoFILT Filtering 

The file `tracks_all.csv` can be filtered using **ExoFILT** via the neural network inference notebook (see below).

#### 4) Manual Annotation 

After ExoFILT filtering, manual annotation can be performed.

Run:
```
3_Annotation_GUI.py
```
The GUI sequentially displays each track (movie crop and intensity profile), allowing the user to assign a label.

#### 5) Colocalization Analysis
If desired, colocalization of bona fide C1 events with channel 2 can be assessed using: 
```
4_Colocalization_GUI.py
```

## Neural Network Inference

### 1. Launch Jupyter

```bash
jupyter lab
```
or
```bash
jupyter notebook
```

### 2. Run the inference notebook

Open:
```
notebooks/inference.ipynb
```
The notebook requires:
  * Preprocessed movies in `3_Preprocessed`
  * A CSV file containing tracks to filter (e.g. `tracks_all.csv` generated by `2_Tracking.py`).

### Inference Workflow

Briefly, the notebook performs the following steps:

1) Apply a permissive parametric filter to the tracks
2) Crop the filtered tracks from the movies (10x10 pixels videos)
3) Load the cropped tracks
4) Perform data augmentation
5) Run inference using five ExoFILT models
6) Compute a final **ExoFILT score**

Users can then select different thresholds on the ExoFILT score to obtain subsets of tracks for downstream manual annotation.

### Notes
  * The pipeline assumes the directory structure generated by the preprocessing scripts.
  * Paths inside the Jupyter Notebook should be adjusted if data are stored in different locations.




