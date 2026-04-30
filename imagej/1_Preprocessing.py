"""
Script:     Preprocessing pipeline for dual-color simultaneous live-cell imaging
Version:    3.0.1
Author:     Eric Kramer i Rosado

Description
-----------
This script preprocesses dual-color simultaneous live-cell imaging datasets. 

Pipeline
-----------
1) Split raw movies into C1 and C2.
2) (Optional) Chromatic alignment of C1 to C2 using bead calibration.
3) Background subtraction (rolling ball).
4) Bleach correction (exponential fit).
5) Export split and preprocessed stacks.

Requirements
------------
- Fiji / ImageJ
- Bio-Formats
- Detection of Molecules (DoM) plugin (for bead correction)
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import os
import glob
from ij                 import IJ, WindowManager
from ij.io              import Opener
from java.lang          import System
from fiji.util.gui      import GenericDialogPlus

#@ PrefService prefs 

# ------------------------------------------------------------------------------
# Splitting
# ------------------------------------------------------------------------------

def split_image(imp, axis, c1_position):
    """
    Split the input image into two halves.
    For vertical splitting, the left/right halves are selected.
    For horizontal splitting, the top/bottom halves are selected.
    """
    
    if imp is None:
        raise ValueError("No image detected for splitting.")
        
    width, height = imp.getWidth(), imp.getHeight()
    original_title = imp.getTitle()
    
    def duplicate_roi(w, h, x, y, title):
        IJ.run(imp, "Specify...",
               "width=%d height=%d x=%d y=%d slice=1 stack" % (w, h, x, y))
        IJ.run("Duplicate...", "title=%s duplicate" % title)

    if axis == "vertical":
        if c1_position not in ["left", "right"]:
            raise ValueError("c1_position must be 'left' or 'right' for vertical splitting.")

        wo = width % 2
        width_half = int(round(width / 2.0)) - wo

        # First half
        first_x = wo
        title = "C1" if c1_position == "left" else "C2"
        duplicate_roi(width_half, height, first_x, 0, title)
        
        # Reselect the original image; ensure it still exists.
        if WindowManager.getImage(original_title) is None:
            raise RuntimeError("Original image lost after first crop.")
        IJ.selectWindow(original_title)
        
        # Second half
        second_x = width_half
        title = "C2" if c1_position == "left" else "C1"
        duplicate_roi(width_half, height, second_x, 0, title)
        
    elif axis == "horizontal":
        if c1_position not in ["top", "bottom"]:
            raise ValueError("c1_position must be 'top' or 'bottom' for horizontal splitting.")

        ho = height % 2
        height_half = int(round(height / 2.0)) - ho
        
        # First half
        first_y = ho
        title = "C1" if c1_position == "top" else "C2"
        duplicate_roi(width, height_half, 0, first_y, title)
        
        # Reselect the original image; ensure it still exists.
        if WindowManager.getImage(original_title) is None:
            raise RuntimeError("Original image lost after first crop.")
        IJ.selectWindow(original_title)

        # Second half
        second_y = height_half
        title = "C2" if c1_position == "top" else "C1"
        duplicate_roi(width, height_half, 0, second_y, title)
        
    else:
        raise ValueError("Axis must be 'vertical' or 'horizontal'.")

    imp_c1 = WindowManager.getImage("C1")
    imp_c2 = WindowManager.getImage("C2")
    
    if imp_c1 is None or imp_c2 is None:
        raise RuntimeError("Splitting failed: C1 or C2 not found.")
    
    imp.hide()
    imp_c1.hide()
    imp_c2.hide()
    
    return imp_c1, imp_c2

# ------------------------------------------------------------------------------
# Chromatic Alignment
# ------------------------------------------------------------------------------

def apply_bead_correction(path_calibration, imp_c1):
    """
    Apply chromatic bead correction to C1.
    """
    
    # Check calibration file exists.
    if not os.path.exists(path_calibration):
        raise RuntimeError("Chromatic calibration file not found: %s" % path_calibration)
    
    # Load the chromatic calibration file
    IJ.run("Load chromatic calibration", "open=[%s]" % path_calibration)
    
    # Record open images BEFORE correction
    titles_before = set(WindowManager.getImageTitles())

    # Apply correction to C1
    print "\tRunning stack correction on C1..."
    IJ.run(imp_c1, "Image or stack correction", "")
    
    # Record open images AFTER correction
    titles_after = set(WindowManager.getImageTitles())
    
    # Detect newly created images
    new_titles = list(titles_after - titles_before)

    if len(new_titles) == 1:
        corrected = WindowManager.getImage(new_titles[0])
    else:
        raise RuntimeError("Unable to uniquely identify corrected image.\n"
                "These candidates were found: %s\n"
                % (new_titles)
            )

    if corrected is None:
        raise RuntimeError("Chromatic correction failed.")

    corrected.setDimensions(1, 1, corrected.getStackSize())
    corrected.setOpenAsHyperStack(True)
    
    return corrected
    
# ------------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------------

def preprocess(input_path, output_path, rolling_radius = 60):
    """
    Preprocess a splitted image:
      - Open the image.
      - Apply background subtraction (rolling ball) and bleach correction.
      - Set image properties (frames, pixel dimensions).
      - Save the preprocessed image.
    """
    IJ.run("Close All", "")
    if not os.path.exists(input_path):
        raise RuntimeError("Input file for preprocessing not found: %s" % input_path)

    # Use the filename as a unique title
    title = os.path.basename(input_path)
    imp = IJ.openImage(input_path)
    imp.setTitle(title)
    imp.show()
    
    # Apply background subtraction
    IJ.run(imp, "Subtract Background...", "rolling=%d stack" % rolling_radius)
    imp_BS = WindowManager.getImage(title)
    
    # Apply bleach correction
    IJ.run(imp_BS, "Bleach Correction", "correction=[Exponential Fit]")
    title_dup = "DUP_" + title
    imp_preprocessed = WindowManager.getImage(title_dup)
        
    num_frames = imp_preprocessed.getNFrames()
    if num_frames < 1:
        print("Warning: Number of frames detected is less than 1.")
        num_frames = 1  # Fallback to 1 to avoid errors.
    elif num_frames == 1:
        print("Warning: Number of frames detected is 1.")
    
    IJ.run(imp_preprocessed, "Properties...", "channels=1 slices=1 frames=%d pixel_width=1 pixel_height=1 voxel_depth=1.0000000" % num_frames)
    IJ.saveAs(imp_preprocessed, "Tiff", output_path)
    IJ.run("Close All", "")

# ------------------------------------------------------------------------------
# Filename Parsing
# ------------------------------------------------------------------------------

def parse_filename(filename_noext, mode):
    """
    Parse experiment name and file_id from filename.

    Supported formats:
    AUTO:
        <experiment>_1_MMStack_Pos<file_id>
    MANUAL:
        <experiment>_<file_id>_MMStack_Pos0
    """
    parts = filename_noext.split("_")

    if mode == "auto":
        parts = filename_noext.split("_")
        pos = parts[-1]  # Expected pattern: ..._1_MMStack_Pos<file_id>
        file_id = pos[3:]  # Remove 'Pos'
        length_filename = len(filename_noext) - 14 - len(file_id)
        experiment = filename_noext[:length_filename]
    elif mode == "manual":
        parts = filename_noext.split("_")
        file_id = parts[-3]  # Expected pattern: ..._<file_id>_MMStack_Pos0
        length_filename = len(filename_noext) - 13 - len(file_id) - 1
        experiment = filename_noext[:length_filename]
    else:
        raise ValueError("Invalid naming mode.")

    return experiment, file_id


# ------------------------------------------------------------------------------
# Main Processing Loop
# ------------------------------------------------------------------------------


def process_folder(input_folder, output_splitted, output_preprocessed, naming_mode, axis, c1_position, path_calibration, align):
    """
    Loop through all .tif files in input_folder.
    For each file:
      - Determine file naming parts based on naming_mode ("auto" or "manual").
      - Open the image, set calibration, and adjust properties.
      - Split the image channels.
      - Run beads correction (optional).
      - Preprocess each channel.
    """
    IJ.run("Close All", "")
    
    if not os.path.exists(input_folder):
        raise RuntimeError("Input folder does not exist: %s" % input_folder)
    
    files = [f for f in sorted(os.listdir(input_folder)) if f.endswith(".ome.tif")]
    
    print "\n*******************************"
    print "Number of files to process:%d" % (len(files))
    print "*******************************\n"
    
    if not files:
        print "No files matching the criteria found in %s" % input_path
        return
    
    for i, filename in enumerate(files):

        print "\nProcessing file %d/%d: %s" %(i+1, len(files), filename)
        filename_noext = filename[:-len(".ome.tif")]
        
        try:
            experiment, file_id = parse_filename(filename_noext, naming_mode)
            print "EXPERIMENT NAME:", experiment, "-- FILE_ID:", file_id
        except Exception as e:
            print("Filename parsing failed:", e)
            continue
        
        path_movie = os.path.join(input_folder, filename)
        imp = Opener.openUsingBioFormats(path_movie)
        if imp is None:
            print "Failed to open: %s" % path_movie
            continue
            
        imp.show()
        
        # Set calibration and properties
        imp.getCalibration().setXUnit("-")
        num_frames = imp.getNFrames()
        IJ.run(imp, "Properties...", "channels=1 slices=1 frames=%d pixel_width=1 pixel_height=1 voxel_depth=1.0000000" % num_frames)
        
        try:
            imp_c1, imp_c2 = split_image(imp, axis, c1_position)
            print "\tSplitting done"
        except Exception as e:
            print("Splitting failed:", e)
            IJ.run("Close All", "")
            continue

        # Build filenames for splitted and preprocessed images
        name_splitted_c1 = "%s_splitted_C1_%s.tif" % (experiment, file_id)
        name_splitted_c2 = "%s_splitted_C2_%s.tif" % (experiment, file_id)
        name_prepro_c1  = "%s_prepro_C1_%s.tif" % (experiment, file_id)
        name_prepro_c2  = "%s_prepro_C2_%s.tif" % (experiment, file_id)
        
        path_splitted_c1 = os.path.join(output_splitted, name_splitted_c1)
        path_splitted_c2 = os.path.join(output_splitted, name_splitted_c2)
        path_prepro_c1 = os.path.join(output_preprocessed, name_prepro_c1)
        path_prepro_c2 = os.path.join(output_preprocessed, name_prepro_c2)

        try:
            if align:
                corrected_c1 = apply_bead_correction(path_calibration, imp_c1)
                IJ.saveAs(corrected_c1, "Tiff", path_splitted_c1)
            else:
                IJ.saveAs(imp_c1, "Tiff", path_splitted_c1)

            IJ.saveAs(imp_c2, "Tiff", path_splitted_c2)
            
        except Exception as e:
            print("Alignment/saving failed:", e)
            IJ.run("Close All", "")
            continue

        try:
            print "\tPreprocessing C1..."
            preprocess(input_path=path_splitted_c1, output_path=path_prepro_c1)
        except Exception as e:
            print "Error during preprocessing for file '%s': %s" % (filename, e)
        try:
            print "\tPreprocessing C2..."
            preprocess(input_path=path_splitted_c2, output_path=path_prepro_c2)
        except Exception as e:
            print "Error during preprocessing for file '%s': %s" % (filename, e)

    print("\nAll files processed.")
    IJ.run("Close All", "")

# ------------------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------------------

# Create a dialog for the user to enter parameters
gui = GenericDialogPlus("Preprocessing Parameters")
gui.addDirectoryField("Root folder:", prefs.get(None, "root_folder", IJ.getDirectory("home")))
gui.addChoice("axis:", ["vertical", "horizontal"], prefs.get(None, "axis", "vertical"))
gui.addToSameRow()
gui.addMessage("direction of the splitting between C1 and C2")
gui.addChoice("c1_position:", ["left", "right", "top", "bottom"], prefs.get(None, "c1_position", "left"))
gui.addToSameRow()
gui.addMessage("position of C1")
gui.addChoice("naming_mode:", ["auto", "manual"], prefs.get(None, "naming_mode", "manual"))
gui.addToSameRow()
gui.addMessage("auto: experiment + _1_MMStack_Pos + file_id + .ome.tif \nmanual: experiment + _ + file_id + _MMStack_Pos0 + .ome.tif")
gui.addCheckbox("Align C1-C2 with beads", bool(prefs.getInt(None, "align", True)))
gui.showDialog()

if gui.wasCanceled():
    print("User canceled dialog")
    exit()
    
root_folder     = gui.getNextString()
axis            = gui.getNextChoice()
c1_position     = gui.getNextChoice()
naming_mode     = gui.getNextChoice()
align           = gui.getNextBoolean()

prefs.put(None, "root_folder", root_folder)
prefs.put(None, "axis", axis)
prefs.put(None, "c1_position", c1_position)
prefs.put(None, "naming_mode", naming_mode)
prefs.put(None, "align", align)

# ------------------------------------------------------------------------------
# Input
# ------------------------------------------------------------------------------

print "\n*******************************"
print "Root folder:", root_folder
print "*******************************"

input_raw = os.path.join(root_folder, "1_Raw")
output_splitted = os.path.join(root_folder, "2_Splitted")
output_preprocessed = os.path.join(root_folder, "3_Preprocessed")

# Create output directories if they do not exist
if not os.path.exists(output_splitted):
    os.makedirs(output_splitted)
if not os.path.exists(output_preprocessed):
    os.makedirs(output_preprocessed)


# Define paths relative to the selected root folder
if align:
    print "\n*******************************"
    print "Alignment will be performed"
    print "*******************************"
    path_calibration = os.path.join(root_folder, "0_Beads", "chromatic_calibration.txt")
    # Check for plugin DoM installation
    ij_dir = System.getProperty("ij.dir")  # Get Fiji's installation folder
    matching_plugins = glob.glob(os.path.join(ij_dir, "plugins", "DoM_*.jar"))
    
    if not matching_plugins:
        raise RuntimeError(
            "Error: The required plugin 'Detection of Molecules' (DoM) is missing from Fiji's plugins folder.\n"
            "Please install it from: https://github.com/UU-cellbiology/DoM_Utrecht."
        )
else:
    print "\n*******************************"
    print "Alignment will NOT be performed"
    print "*******************************"
    path_calibration = None
    
# ------------------------------------------------------------------------------
# Run everything
# ------------------------------------------------------------------------------

process_folder(input_raw, output_splitted, output_preprocessed, naming_mode, axis, c1_position, path_calibration, align)

