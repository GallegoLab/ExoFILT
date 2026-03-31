<h1 align="center">ExoFILT</h1>
<h3 align="center">Transfer learning for robust and accelerated analysis of exocytosis single-particle tracking data</h3>

**ExoFILT** is a deep learning binary classifier designed to identify **bona fide exocytic events** from live-cell imaging data in *Saccharomyces cerevisiae*.

For more details about this method, see [ExoFILT: Transfer learning for robust and accelerated analysis of exocytosis single-particle tracking data](https://www.biorxiv.org/content/10.64898/2026.02.27.708581v1)

## Overview

This repository provides a complete pipeline for analyzing exocytosis events from live-cell imaging data. It consists of three main components:

1. **ImageJ/Fiji scripts**  
   Tools for preprocessing microscopy movies, performing automated tracking, and manually annotating tracks.
   
2. **Neural network inference**  
   A Jupyter notebook that applies **ExoFILT** to classify tracked exocytic events.
   
3. **Data visualization app**  
   An interactive app based on the Streamlit package to visualize results from dual-color live-cell imaging experiments.

Pretrained ExoFILT models used for inference are provided in the repository under the `models/` directory.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/GallegoLab/ExoFILT.git
cd ExoFILT
```

### 2. Create and activate the Conda environment

Two environments are provided, one for easy installation (using only CPU) and one for fast neural network inference using GPU. 

#### CPU Environment (recommended for most users)

Create the conda environment and install the required packages:
```bash
conda env create -f environment_cpu.yml
```

After installation, activate the environment:
```bash
conda activate exofilt_cpu
```

#### GPU Environment (optional, faster inference)

If a GPU is available, follow these steps to increase the inference speed.
```bash
conda env create -f environment_gpu.yml
```

After installation, activate the environment:
```bash
conda activate exofilt_gpu
```

The following step ensures that TensorFlow can correctly locate GPU libraries inside the Conda environment:
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

## Quick Start

1. Run preprocessing in ImageJ (`1_Preprocessing.py`)
2. Run tracking in ImageJ (`2_Tracking.py`)
3. Run neural network inference (`notebooks/inference.ipynb`)
4. Manually curate results (`3_Annotation_GUI.py`)

## ImageJ / Fiji Preprocessing & Annotation

### Requirements
The ImageJ scripts were developed and tested with:
  * ImageJ 1.54p
  * Java 1.8.0_322
  * TrackMate v7.14.0
  * Jython scripting enabled
  * Accurate Gaussian Blur plugin. 
  
A standard **Fiji** installation should contain the required dependencies, except `Accurate Gaussian Blur` plugin.
To install it, download [Accurate_Gaussian_Blur.class](https://imagej.net/ij/plugins/download/Accurate_Gaussian_Blur.class), 
copy to the plugins folder, and restart ImageJ. 
  
### Scripts

1) `1_Preprocessing.py`: Preprocess simultaneous dual-color live-cell imaging movies.
2) `2_Tracking.py`: Automated particle tracking of preprocessed movies using TrackMate.
3) `3_Annotation_GUI.py`: GUI for manual annotation of individual tracks in channel 1.
4) `4_Colocalization_GUI.py`: GUI for manual annotation of colocalization between channels 1 (C1) and 2 (C2).
5) `5_Extract_Intensity.py`: Extract intensity profiles for each pair of C1-C2 bona fide events.

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

After selecting the appropriate parameters in the initial dialog, the script creates two new directories:
```
2_Splitted
3_Preprocessed
```
These folders contain channel-separated movies, before and after preprocessing, respectively.

#### 2) Automated Tracking
Open `2_Tracking.py` in ImageJ and run it with Jython. 
 
The script performs automated tracking using **TrackMate** on the C1 movies located in `3_Preprocessed`.

All tracking outputs are written to the folder `4_Tracking`. The main file containing all tracks is saved as `tracks_all.csv`.

#### 3) ExoFILT Filtering 
The file `tracks_all.csv` can be filtered using **ExoFILT** through the neural network inference notebook (see below).

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
Briefly, the GUI takes as input a CSV file with bona fide C1 events manually annotated (e.g., selected from the output in the previous step). 
It opens the C1 bona fide event alongside the same region in C2, both from the preprocessed movies. After visual inspection, if a corresponding 
event is detected in C2, TrackMate can be run on C2 and the resulting data stored. Each pair of C1-C2 tracks is given a single `COLOCALIZE_ID`. 
TrackMate data of each pair of tracks is stored in a new CSV file.

A new folder (`5_Analysis`) is created, where data required by the next steps is stored.

#### 6) Extract Intensity Profiles
To prepare the data for further analysis with the **Data Visualization app**, the following script should be run in ImageJ:
```
5_Extract_Intensity.py
```
Briefly, this script takes as input a CSV file with C1-C2 bona fide pairs (selected from the output of `4_Colocalization_GUI.py`). By checking the exact
coordinates of each event through the movie (Spot data found in folders `4_Tracking` and `5_Analysis`), it extracts the raw intensity data of the spot 
from the splitted movies (`2_Splitted`). A local background subtraction is also applied. Both raw and corrected intensity profiles are stored together 
(one file per track and per channel) inside the folder `5_Analysis`.

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
  * Preprocessed movies in `3_Preprocessed` (generated by `1_Preprocessing.py`, see above).
  * A CSV file containing tracks to filter (e.g. `tracks_all.csv` generated by `2_Tracking.py`).

### Inference Workflow

Briefly, the notebook performs the following steps:

1) Apply a permissive parametric filter to the tracks. By default, the filter defined in the ExoFILT study is used.
2) Crop the filtered tracks from the movies (10 x 10 pixels videos). A folder named `crops_raw` is generated inside `4_Tracking`, 
and can be deleted after the analysis.
3) Load the cropped tracks.
4) Perform data augmentation.
5) Run inference using five ExoFILT models. A file with the mean score from each model is stored as `summary_NN_preds.csv` in 
a new folder named `NN_predictions` inside `4_Tracking`.
6) Compute a final **ExoFILT score** by averaging the scores from the five models. Users can then select different thresholds on the ExoFILT 
score to obtain subsets of tracks for downstream manual annotation (using `3_Annotation_GUI.py` in ImageJ, see above).

### Notes
  * The pipeline assumes the directory structure generated by the preprocessing scripts.
  * Paths inside the Jupyter Notebook should be adjusted if data is stored in different locations.

## Data visualization App

An interactive app built with **Streamlit** enables exploration and visualization of the results from multiple 
simultaneous dual-color live-cell imaging experiments. When combining multiple experiments, the pipeline assumes
that the protein in C1 is consistent across experiments and is used as a reference.

Four main types of plots are generated:
  * Individual timelines: Display the timeline (start, stop, and duration) of each individual pair of C1-C2 bona fide events in each experiment.
  * Average timelines: Display the average timeline for a whole dataset.
  * Individual intensity profiles: Display a mosaic of intensity profiles for each individual pair of C1-C2 bona fide events.
  * Average intensity profiles: Display the average intensity profile for each experiment.

The interactive app can be opened in a browser with the following command:
```bash
streamlit run data_visualization/app.py
```

For each experiment to be analyzed, two input files are required:
  * CSV with TrackMate data on pairs of C1-C2 bona fide events.
  * Intensity profiles in a .zip file (compress manually the folder generated by `5_Extract_Intensity.py`).

