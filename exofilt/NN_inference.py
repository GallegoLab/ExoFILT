# ---------------------------
# Standard libraries
# ---------------------------
import os
import time
import string
import shutil
import warnings

# Silence TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore")

# ---------------------------
# Scientific Python
# ---------------------------
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import skimage

# ---------------------------
# Deep learning
# ---------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Allow importing project modules
import sys
sys.path.append("..")

##################################################
# Tensorflow: GPU/CPU settings
##################################################

def configure_tensorflow():
    """
    Configure the TensorFlow execution environment.
    
    This function clears any existing TensorFlow/Keras session and configures
    GPU usage if available. When GPUs are detected, dynamic memory growth is
    enabled to prevent TensorFlow from allocating all GPU memory at startup.
    
    Notes
    -----
    Prints whether TensorFlow is running on GPU or CPU.
    """
    
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow using GPU ({len(gpus)} GPUs detected)")
        except RuntimeError as e:
            print(f"GPU setup failed: {e}")
    else:
        print("GPU not detected. TensorFlow using CPU")

##################################################
# PREPROCESSING: PERMISSIVE FILTER + CROPPING
##################################################

def filter_df(data, filters, verbose=False):
    """
    Filter a DataFrame based on threshold ranges for specified columns.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame containing track features.
    filters : dict
        Dictionary defining filtering thresholds:
        {column_name: (lower_bound, upper_bound)}.
    verbose : bool, optional
        If True, print filtering statistics.
    
    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only rows satisfying all filters.
    """
    
    # Start with a mask that keeps everything
    mask = pd.Series(True, index=data.index)

    # Apply filters based on thresholds
    for column, (lower, upper) in filters.items():
        
        if column not in data.columns:
            warnings.warn(f"Column '{column}' not found in DataFrame. Skipping filter.")
            continue
        
        mask &= (data[column] >= lower) & (data[column] <= upper)
    
    data_filtered = data[mask].copy()

    if verbose:
        print("\n******* APPLYING PARAMETRIC FILTER *******")
        print("File_ID\t\tFiltered\tTotal")

        original_counts = data["FILE_ID"].value_counts().sort_index()
        filtered_counts = data_filtered["FILE_ID"].value_counts().sort_index()

        for file_id in original_counts.index:
            total = original_counts[file_id]
            filtered = filtered_counts.get(file_id, 0)
            print(f"{file_id}\t\t{filtered}\t\t{total}")

        total_filtered = len(data_filtered)
        total_original = len(data)
        percentage = round(100 * total_filtered / total_original, 2)
        print(f"\nTotal filtered tracks: {total_filtered}/{total_original} ({percentage}%)")

    return data_filtered

def crop_tracks_from_df(df_tracks, path_wholeFOV, path_crops, 
                        size_crop = 10, extra_padding = 18):
    
    """
    Crop square regions around tracks from whole videos.

    For each track in `df_tracks`, a spatiotemporal crop is extracted from
    the corresponding whole-field movie and saved as a `.npy`file. Crops 
    include additional frames before and after the track to provide temporal 
    context.
    
    Parameters
    ----------
    df_tracks : pandas.DataFrame
        DataFrame containing tracking information. Must include the columns:
        ['EXPERIMENT', 'FILE_ID', 'TRACK_START', 'TRACK_STOP',
         'TRACK_X_LOCATION', 'TRACK_Y_LOCATION'].
    path_wholeFOV : str
        Path to the directory containing the full-field movies (.tif files).
    path_crops : str
        Directory where cropped tracks will be saved.
    size_crop : int, optional
        Spatial size of the square crop in pixels. Default is 10.
    extra_padding : int, optional
        Number of frames added before and after the track interval. Default is 18.
    """
    
    path_wholeFOV = os.path.abspath(path_wholeFOV)
    path_crops = os.path.abspath(path_crops)
    os.makedirs(path_crops, exist_ok=True)
    
    print("\n******* CROPPING *******")
    print(f"Processing {len(df_tracks)} tracks")
    print(f"Whole FOV path: \n\t{path_wholeFOV}")
    print(f"Crops will be saved here: \n\t{path_crops}\n")

    experiments = df_tracks["EXPERIMENT"].unique()
    if len(experiments) != 1:
        raise ValueError("df_tracks must contain a single EXPERIMENT")
    experiment_name = experiments[0]

    half_size = size_crop // 2

    # Iterate over each unique file ID in the DataFrame
    for file_id, subset_tracks in df_tracks.groupby("FILE_ID"):
        print(f"Processing FILE_ID {file_id} | {len(subset_tracks)} tracks")
        
        filename = f"{experiment_name}_prepro_C1_{str(file_id)}.tif"
        filepath = os.path.join(path_wholeFOV, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found in {path_wholeFOV}")
 
        # Load the full movie
        full_array = tifffile.imread(filepath)  # Shape: (frames, Y, X)
        n_frames, y_max, x_max = full_array.shape
        
        for row in subset_tracks.itertuples(index=False):
            start = int(row.TRACK_START)
            stop = int(row.TRACK_STOP)
            x = int(round(row.TRACK_X_LOCATION))
            y = int(round(row.TRACK_Y_LOCATION))

            # Compute safe frame indices
            f_start = max(0, start - extra_padding)
            f_stop = min(n_frames, stop + extra_padding)

            # Compute safe spatial indices
            y_start = y - half_size
            y_stop = y + half_size
            x_start = x - half_size
            x_stop = x + half_size
            
            # Skip tracks near borders
            if y_start < 0 or x_start < 0 or y_stop > y_max or x_stop > x_max:
                print("Track to close to border, crop not possible. Track skipped")
                continue
            
            crop = full_array[f_start:f_stop, y_start:y_stop, x_start:x_stop]
            
            crop_filename = f"{experiment_name}_file_{file_id}_crop_X_{x}_Y_{y}_from_{start}_to_{stop}"

            npy_path = os.path.join(path_crops, f"{crop_filename}.npy")
            np.save(npy_path, crop)

    print("Done with all files")

##################################################
# CREATE DATA GENERATOR
##################################################

class InferenceDataGenerator(keras.utils.Sequence):
    """
    Keras Sequence generator for inference.
    
    This generator loads cropped track videos stored as `.npy` files and
    applies augmentation to generate multiple transformed versions of each
    video. The augmented versions are returned as input for neural network
    prediction.
    
    Parameters
    ----------
    filenames : list of str
        List of crop filenames.
    path : str
        Directory containing the `.npy` crop files.
    augment_video : callable
        Function that generates augmented versions of a video.
    batch_size : int, optional
        Number of samples per batch. Default is 1.
    
    Notes
    -----
    Each sample returned by the generator corresponds to a set of augmented
    versions of the same crop.
    
    """
    
    def __init__(self, filenames, path, augment_video, batch_size=1):
        self.filenames = filenames
        self.path = path
        self.augment_video = augment_video
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, idx):
        
        filename = self.filenames[idx]
        filepath = os.path.join(self.path, filename)
        
        original_video = np.load(filepath)

        augmented_versions = self.augment_video(original_video)

        return augmented_versions

def augment_video(array):
    """
    Generate augmented versions of a video.
    
    The input video is first normalized and smoothed using a Gaussian filter.
    Six augmented versions are then produced:
    the original, three rotations (90°, 180°, 270°), and two mirrored versions.
    
    Parameters
    ----------
    array : numpy.ndarray
        Input video array with shape (frames, height, width).
    
    Returns
    -------
    numpy.ndarray
        Array containing augmented videos with shape:
        (6, frames, height, width).
    """
    
    # Normalize video
    array_min = np.amin(array)
    array_max = np.amax(array)
    
    if array_max > array_min:
        array_norm = (array - array_min) / (array_max - array_min) # Scale between 0 and 1
    else:
        array_norm = np.zeros_like(array)

    # Apply the Gaussian filter to the 3D array
    sigma_gaussian = (1.5, 0.5, 0.5)
    array_norm_gaussian = skimage.filters.gaussian(array_norm, sigma = sigma_gaussian)

    augmented_versions = [array_norm_gaussian] # original version ("rot = 0")

    for k in [1,2,3]:
        array_rotated = np.rot90(array_norm_gaussian, k = k, axes = (1, 2))
        augmented_versions.append(array_rotated)
    for axis in [1,2]:
        array_mirrored = np.flip(array_norm_gaussian, axis = axis)
        augmented_versions.append(array_mirrored)

    return np.stack(augmented_versions)  # Shape: (6, length, height, width)

def create_generator(path_crops, with_dummy = False):
    """
    Create a data generator for neural network inference.
    
    The generator loads cropped `.npy` files and produces augmented
    versions of each crop for prediction.
    
    Parameters
    ----------
    path_crops : str
        Directory containing cropped track files (.npy).
    with_dummy : bool, optional
        If True, also return a smaller generator containing a subset
        of up to 300 samples. Useful for quick testing.
    
    Returns
    -------
    InferenceDataGenerator or tuple
        If `with_dummy` is False, returns the full generator.
        If True, returns a tuple (generator, generator_dummy).
    """
    
    # List all cropped .npy files and sort them
    filename_crops = sorted([f for f in os.listdir(path_crops) if f.endswith(".npy")])
    
    if not filename_crops:
        raise ValueError(f"No .npy crops found in {path_crops}")
        
    generator = InferenceDataGenerator(
        filenames = filename_crops, 
        path = path_crops, 
        augment_video = augment_video, 
        batch_size = 1
    )

    if with_dummy:
        subset = filename_crops[0: min(300, len(filename_crops))]
        generator_dummy = InferenceDataGenerator(
            filenames = subset, 
            path = path_crops, 
            augment_video = augment_video,
            batch_size = 1
        )
        return generator, generator_dummy
    else:
        return generator

##################################################
# MAKE PREDICTIONS
##################################################

def predict_single_model(generator, path_model, num_augmentations=6, plot = True, dpi = 200):
    """
    Compute predictions for a single neural network model.
    
    Each crop is evaluated using multiple augmented versions. Predictions
    for the augmented versions are averaged to obtain a single score
    per crop.
    
    Parameters
    ----------
    generator : InferenceDataGenerator
        Data generator providing augmented crops.
    path_model : str
        Path to the trained Keras model.
    num_augmentations : int, optional
        Number of augmented versions per crop. Default is 6.
    plot : bool, optional
        If True, display a histogram of the averaged prediction scores.
        Default is True.
    dpi : int, optional
        Resolution of the plot. Default is 200.
    
    Returns
    -------
    dict
        Dictionary mapping crop filenames to their averaged prediction score.
    
    Notes
    -----
    Keras concatenates predictions from augmented samples, so the total
    number of predictions equals: `N_crops × num_augmentations`.
    """
        
    # Load the pre-trained model
    model = load_model(path_model)

    # Compute predictions
    predictions = model.predict(generator, verbose = 1)
    # Each generator item returns `num_augmentations` videos.
    # Keras concatenates them, so predictions length = N_crops * num_augmentations.
    
    filenames = generator.filenames
    if len(predictions) != len(filenames) * num_augmentations:
        raise ValueError("Mismatch between predictions and expected augmented crops length.")
    
    # Group predictions per crop
    preds_per_crop = [predictions[i:i+num_augmentations] for i in range(0, len(predictions), num_augmentations)]
    
    # Map filename -> list of predictions
    filename_to_preds = dict(zip(filenames, preds_per_crop))
    
    # Compute mean prediction per crop
    mean_preds = {filename: np.mean(preds) for filename, preds in filename_to_preds.items()}
    
    # Optional plotting
    if plot:
        plt.style.use("default")
        fig, ax = plt.subplots(1, 1, figsize = (3,2), dpi = dpi)
        ax.hist(list(mean_preds.values()), edgecolor = "black", bins = 8)
        ax.set_xlim(-0.1,1.1)
        ax.set_xlabel(f"Average score over {num_augmentations} augmentations")
        plt.axvline(x = 0.5, linestyle="--", color = "gray", alpha = 0.5)
        model_name = os.path.basename(path_model)
        fig.suptitle(f"Model {model_name}", fontsize=8)
        plt.show()
    
    return mean_preds

def predict_multiple_models(generator, model_names, path_models, plot = True, dpi = 200):
    """
    Compute predictions using multiple neural network models.
    
    For each model, predictions are generated and averaged across
    augmented versions of each crop. Results from all models are
    combined into a single DataFrame.
    
    Parameters
    ----------
    generator : InferenceDataGenerator
        Data generator providing augmented crops.
    model_names : list of str
        Names of the models to evaluate.
    path_models : str
        Directory containing the trained models.
    plot : bool, optional
        If True, display a histogram of prediction scores for each model.
        Default is True.
    dpi : int, optional
        Resolution of the plot. Default is 200.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing prediction scores for each crop.
        Columns include:
        ["Sample", "Mean preds <model_1>", ...].
    """
    
    # Initialize dictionary keyed by filename
    all_preds_dict = {filename: {} for filename in generator.filenames}
    
    for model_name in model_names:
        print(f"Using model: {model_name}")
        path_model = os.path.join(path_models, model_name)
        
        mean_preds = predict_single_model(generator, path_model, plot = plot, dpi = dpi)

        # Store predictions in dictionary
        for filename, mean_pred in mean_preds.items():
            all_preds_dict[filename][f"Mean preds {model_name}"] = mean_pred

    # Convert dictionary to DataFrame
    df_rows = []
    for filename, preds_dict in all_preds_dict.items():
        row = {"Sample": filename}
        row.update(preds_dict)
        df_rows.append(row)
    
    df_preds = pd.DataFrame(df_rows)
    return df_preds

##################################################
# PROCESS PREDICTIONS
##################################################

def subset_by_avg_prediction(df_preds, threshold, plot = False, dpi = 200):
    """
    Select crops based on the average prediction score across models.
    
    The function computes the mean prediction across all models and
    splits the dataset into two subsets based on a threshold.
    
    Parameters
    ----------
    df_preds : pandas.DataFrame
        DataFrame containing prediction scores from multiple models.
    threshold : float
        Threshold applied to the average prediction score.
    plot : bool, optional
        If True, display a histogram of average prediction scores.
        Default is False.
    dpi : int, optional
        Resolution of the plot. Default is 200.
    
    Returns
    -------
    tuple of pandas.DataFrame
        df_selected : crops with average score below the threshold.
        df_remaining : crops with average score above or equal to the threshold.
    """
    
    mean_cols = [col for col in df_preds.columns if "Mean preds" in col]
    
    # Compute average across models
    df_preds = df_preds.copy()
    df_preds["Average_mean_preds"] = df_preds[mean_cols].mean(axis=1)

    # Subset based on threshold
    df_selected = df_preds[df_preds["Average_mean_preds"] < threshold].sort_values("Average_mean_preds")
    df_remaining = df_preds[df_preds["Average_mean_preds"] >= threshold]

    if plot:
        plt.style.use("default")
        fig, ax = plt.subplots(1, 1, figsize = (3,2), dpi = dpi)
        ax.hist(df_preds["Average_mean_preds"], bins=8, edgecolor='black')
        ax.set_xlim(-0.1,1.1)
        ax.set_xlabel(f"Average score across models (N={len(mean_cols)})")
        plt.show()

    return df_selected, df_remaining
    
def subset_by_multiple_thresholds(df_preds, thresholds):
    """
    Partition predictions into multiple score ranges using thresholds.
    
    Thresholds are applied sequentially to divide predictions into
    intervals (e.g., 0–0.2, 0.2–0.4, etc.).
    
    Parameters
    ----------
    df_preds : pandas.DataFrame
        DataFrame containing prediction scores.
    thresholds : list of float
        List of thresholds defining score intervals.
    
    Returns
    -------
    list of tuples
        List of tuples of the form:
        (start_threshold, end_threshold, df_subset)
        where `df_subset` contains crops within the corresponding range.
    """
    
    df_remaining = df_preds.copy()
    list_df_selected = []
    thresholds = sorted(thresholds)
    
    for i, threshold in enumerate(thresholds):
        if i == 0:
            df_sel, df_remaining = subset_by_avg_prediction(df_remaining, threshold, plot = True, dpi = 200)
            list_df_selected.append((0.0, threshold, df_sel))
            print(f"Score: 0.00-{threshold} --> {len(df_sel)} crops")
        else:
            df_sel, df_remaining = subset_by_avg_prediction(df_remaining, threshold, plot = False)
            list_df_selected.append((thresholds[i-1], threshold, df_sel))
            print(f"Score: {thresholds[i-1]}-{threshold} --> {len(df_sel)} crops")
            
    return list_df_selected

def map_crops_to_tracks(df_selected, df_alltracks):
    """
    Map crop filenames back to their corresponding tracks.
    
    This function reconstructs the link between cropped videos and
    their original track entries in the tracking DataFrame.
    
    Parameters
    ----------
    df_selected : pandas.DataFrame
        DataFrame containing selected crops with column "Sample".
    df_alltracks : pandas.DataFrame
        DataFrame containing all tracked events.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the rows from `df_alltracks` corresponding
        to the selected crops.
    
    Notes
    -----
    Crop filenames are reconstructed based on track metadata to
    identify the corresponding rows in `df_alltracks`.
    """
    
    experiments = df_alltracks["EXPERIMENT"].unique()
    if len(experiments) != 1:
        raise ValueError("DataFrame must contain a single EXPERIMENT")
    experiment_name = experiments[0]
    
    # Build lookup dictionary
    dict_crop_to_row = {}
    for row in df_alltracks.itertuples(index=False):
        filenumber = int(row.FILE_ID)
        start = int(row.TRACK_START)
        stop = int(row.TRACK_STOP)
        x = int(round(row.TRACK_X_LOCATION))
        y = int(round(row.TRACK_Y_LOCATION))
        filename_crop = f"{experiment_name}_file_{filenumber}_crop_X_{x}_Y_{y}_from_{start}_to_{stop}.npy"
        dict_crop_to_row[filename_crop] = row
    
    # Collect rows for selected crops
    rows = []
    missing = []

    for filename in df_selected['Sample']:
        if filename in dict_crop_to_row:
            rows.append(dict_crop_to_row[filename])
        else:
            missing.append(filename)
    if missing:
        print(f"Warning: {len(missing)} crops not found in original tracks!")
        
    df_tracks_selected = pd.DataFrame(rows)
    return df_tracks_selected

def save_subsets_by_threshold(list_df_selected, df_alltracks, path_save_subset):
    """
    Save track subsets corresponding to prediction score intervals.
    
    Each subset produced by `subset_by_multiple_thresholds` is mapped
    back to the original tracking DataFrame and saved as a CSV file.
    
    Parameters
    ----------
    list_df_selected : list of tuples
        Output of `subset_by_multiple_thresholds`. Each tuple contains
        (start_threshold, end_threshold, df_subset).
    df_alltracks : pandas.DataFrame
        DataFrame containing all tracked events.
    path_save_subset : str
        Directory where the CSV files will be saved.
    
    Notes
    -----
    Files are named according to the score interval and assigned a
    letter identifier (A, B, C, ...).
    """

    print(f"Saving selected subsets in: {path_save_subset}")
    
    for i, (thr_start, thr_stop, df_subset) in enumerate(list_df_selected):
        
        df_tracks = map_crops_to_tracks(df_subset, df_alltracks)
        
        thr_start_str = str(thr_start).replace(".", "")
        thr_stop_str = str(thr_stop).replace(".", "")
        
        filename = f"NNpreds_{string.ascii_uppercase[i]}_score_{thr_start_str}_{thr_stop_str}.csv"
        print(f"\t- {filename} (N={len(df_tracks)})")
        
        os.makedirs(path_save_subset, exist_ok=True)
        df_tracks.to_csv(os.path.join(path_save_subset, filename), index = False)

