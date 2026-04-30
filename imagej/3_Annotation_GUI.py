"""
Script:    Annotation_GUI
Version:   3.0.1
Author:    Eric Kramer i Rosado

Description
-----------
    Display a GUI for manual annotation of exocytic events. From a list of tracks (csv file),
    display the corresponding video with its intensity profile, and allow the user to annotate 
    each track in a sequential manner.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Standard libraries
import csv
import os
from collections import defaultdict

# ImageJ
import ij
from ij import IJ, WindowManager
from ij.io import Opener
from ij.gui import NonBlockingGenericDialog, Plot, OvalRoi, Overlay
from ij.measure import ResultsTable
from ij.plugin import ImageCalculator

# Fiji utilities
from fiji.util.gui import GenericDialogPlus

# Java classes
from java.awt import Color, FlowLayout, Dimension, Toolkit 
from java.util.concurrent import CountDownLatch

# Swing
from javax.swing import JDialog, JPanel, JRadioButton, JButton, BoxLayout, ButtonGroup

#@ PrefService prefs

########################################################
#################### ALL FUNCTIONS #####################
########################################################

def crop_from_coordinates(whole_FOV, crop_size, x_center, y_center, track_start, track_stop, extra_frames):
    """
    Extract a substack centered on (x_center, y_center) in space and extending
    from (track_start - extra_frames) to (track_stop + extra_frames) in time.

    Returns:
      crop: ImagePlus of the cropped stack
      crop_start, crop_stop: start and stop frame indices (0-based) in the original stack
    """
    if whole_FOV is None:
        raise ValueError("Error: The provided ImagePlus object (whole_FOV) is None.")

    # Get image dimensions
    width, height, Nframes = whole_FOV.getWidth(), whole_FOV.getHeight(), whole_FOV.getNFrames()
    
    # Check that the image has frames
    if Nframes <= 0:
        raise ValueError("Error: The image has no frames.")
    if crop_size > width or crop_size > height:
        raise ValueError("Desired crop size (%d) is larger than the image dimensions (%dx%d)" % (crop_size, width, height))

    # Initial crop bounds (ideally centered on x_center and y_center)
    half_size = crop_size / 2.0
    x0 = int(round(x_center - half_size))
    y0 = int(round(y_center - half_size))
    
    # Initialize crop dimensions
    adjusted_crop_x = crop_size
    adjusted_crop_y = crop_size
    
    # Adjust x limits to keep crop centered
    if x0 < 0:
        # Not enough room on left; maximum size is 2*x_center
        adjusted_crop_x = 2 * x_center
        x0 = 0
        print "Shrink crop width to %d (too close to left edge)" % int(round(adjusted_crop_x))
    elif x0 + crop_size > width:
        # Not enough room on right; maximum size is 2*(width - x_center)
        adjusted_crop_x = 2 * (width - x_center)
        x0 = int(round(width - adjusted_crop_x))
        print "Shrink crop width to %d (too close to right edge)" % int(round(adjusted_crop_x))
        
    # Adjust y limits
    if y0 < 0:
        adjusted_crop_y = 2 * y_center
        y0 = 0
        print "Shrink crop height to %d (too close to top edge)" % int(round(adjusted_crop_y))
    elif y0 + crop_size > height:
        adjusted_crop_y = 2 * (height - y_center)
        y0 = int(round(height - adjusted_crop_y))
        print "Shrink crop height to %d (too close to bottom edge)" % int(round(adjusted_crop_y))
    
    # Final check: ensure adjusted crop sizes are > 0
    if adjusted_crop_x <= 0 or adjusted_crop_y <= 0:
        raise ValueError("Adjusted crop dimensions (%dx%d) are not valid. Check the track coordinates and crop_size." % (adjusted_crop_x, adjusted_crop_y))

    if adjusted_crop_x != crop_size or adjusted_crop_y != crop_size:
        print "Final crop size: %dx%d" % (int(round(adjusted_crop_x)), int(round(adjusted_crop_y)))

    # Adjust time cropping range
    stack = whole_FOV.getStack()
    crop_start = max(0, track_start - extra_frames)
    crop_stop = min(Nframes, track_stop + extra_frames)
    duration = crop_stop - crop_start
    if duration <= 0:
        raise ValueError("Duration of crop is non-positive. Check track frames and extra_frames.")

    # Create crop in both space and time
    crop_stack = stack.crop(x0, y0, crop_start, int(round(adjusted_crop_x)), int(round(adjusted_crop_y)), duration)
    # Give a title to the crop, based on its TrackMate parameters
    crop_title = "crop_X_%d_Y_%d_from_%d_to_%d.tif" % (x_center, y_center, track_start, track_stop)
    # Transform the stack object into an ImagePlus object
    crop = ij.ImagePlus(crop_title, crop_stack)
    
    return crop, crop_start, crop_stop

########################################################

def display_gaussian(raw, track_start, track_stop, crop_start, zoom = 1500, gaussian_blur_parameters = (0.5, 0.5, 1.5), ROI_diameter = 10):
    """
    From a raw video, apply a 3D gaussian blur and combine the original and blurred stacks laterally. 
    The combined image is displayed with the desired zoom.
    
    Parameters:
      raw: raw ImagePlus object.
      zoom (int): Zoom factor for displaying the combined image.
      mark_ROI (bool): If True, draw a 10x10 rectangle in the center of the Gaussian-blurred video.
    """
    
    # Ensure image is provided
    if raw is None:
        raise ValueError("Error: raw image is None.")
        
    # Duplicate raw stack and apply 3D gaussian blur
    raw.show()
    gaussian = raw.duplicate()
    IJ.run(gaussian, "Gaussian Blur 3D...", "x=%.2f y=%.2f z=%.2f" % (gaussian_blur_parameters[0], gaussian_blur_parameters[1], gaussian_blur_parameters[2]))                        
    raw.close()
    
    gaussian.show()
    IJ.run(gaussian, "Set... ", "zoom=%d"%(zoom))
    IJ.run(gaussian, "Enhance Contrast", "saturated=0.35")
    
    
    crop_width, crop_height = gaussian.getWidth(), gaussian.getHeight()
    x_center, y_center = crop_width//2, crop_height//2
    roi_x, roi_y = x_center - ROI_diameter//2, y_center - ROI_diameter//2

    start = track_start - crop_start
    end   = track_stop - crop_start

    overlay = Overlay()
    
    for frame in range(start, end + 1):
        roi = OvalRoi(roi_x, roi_y, ROI_diameter, ROI_diameter)
        roi.setPosition(frame)
        roi.setStrokeColor(Color.GREEN)
        overlay.add(roi)

    gaussian.setOverlay(overlay)
    IJ.run("Tile")
    return gaussian

########################################################

def display_intensity_profile(crop, sigma1, sigma2):
    """
    Compute and display the Difference-of-Gaussians (DoG) intensity profile for a stack. 
    The profile is computed using an oval ROI (fixed radius) at (x_center, y_center) over a time range 
    from track_start-extra_frames to track_stop+extra_frames.
    
    Parameters:
      whole_FOV: ImagePlus object representing the full FOV stack
      x_center, y_center (float): Track center coordinates.
      track_start, track_stop (int): Frame indices for track start and stop.
      extra_frames (int): Number of extra frames to include before and after the detected track.
      dog_sigma1, dog_sigma2 (float): Sigma values for the Gaussian blur.
    """
    # Ensure image is provided
    if crop is None:
        raise ValueError("Error: Crop image is None.")
        
    # Duplicate for DoG computation
    gaussian1 = crop.duplicate()
    gaussian2 = crop.duplicate()
    
    IJ.run(gaussian1, "Gaussian Blur...", "sigma=%.2f stack" % sigma1)
    IJ.run(gaussian2, "Gaussian Blur...", "sigma=%.2f stack" % sigma2)
    
    # Compute DoG image
    dog = ImageCalculator.run(gaussian1, gaussian2, "subtract create 32-bit stack")
    dog.setTitle("DoG_Crop")
    dog.show()
    
    # Define a small oval ROI around the center
    width, height = crop.getWidth(), crop.getHeight()
    x_center, y_center = width/2, height/2
    diameter = 3  # Fixed diameter for intensity measurement
    IJ.makeOval(x_center - diameter/2 + 0.5, y_center - diameter/2 + 0.5, diameter, diameter)
    
    # Measure intensity in the ROI across the stack and plot the intensity profile
    IJ.run("Measure Stack...")
    results_table = ResultsTable.getResultsTable("Results")
    plot_profile = Plot("Spot intensity profile (DoG)", "Frames", "Intensity")
    plot_profile.add("Line", results_table.getColumn("Mean"))
    
    # Close the results window and show the plot
    IJ.selectWindow("Results"); IJ.run("Close")
    
    plot_profile.show()
    plot_profile.setFrozen(True)  # Freeze to allow tiling with other images if desired

    # Clean up
    dog.close(); gaussian1.close(); gaussian2.close()
    IJ.run("Tile")
    
########################################################

def gui_reason_discarded():
    """
    Create a GUI for the user to select the reason(s) for discarding a track.
    Returns a list of selected reasons.
    """
    
    gui = NonBlockingGenericDialog("Select reason for discarding")
    gui.addCheckbox("Passing", False)
    gui.addCheckbox("Neck", False)
    gui.addCheckbox("Bud", False)
    gui.addCheckbox("Dim", False)
    gui.addCheckbox("Out of focus", False)
    gui.addCheckbox("Spot overlapping", False)
    gui.addCheckbox("Big cluster", False)
    gui.addCheckbox("Start/end not shown", False)
    gui.addCheckbox("Too short", False)
    gui.addCheckbox("Center cell", False)
    gui.addCheckbox("Nuf2", False)
    gui.addStringField("Other:", None)
    
    gui.centerDialog(True)
    gui.showDialog()
    
    if gui.wasOKed():
        bool_passing = gui.getNextBoolean()
        bool_neck = gui.getNextBoolean()
        bool_bud = gui.getNextBoolean()
        bool_dim = gui.getNextBoolean()
        bool_oof = gui.getNextBoolean()
        bool_overlap = gui.getNextBoolean()
        bool_bigcluster = gui.getNextBoolean()
        bool_startend = gui.getNextBoolean()
        bool_short = gui.getNextBoolean()
        bool_center = gui.getNextBoolean()
        bool_nuf2 = gui.getNextBoolean()
        other = gui.getNextString()
    else:
        raise Exception("Discard reason dialog was cancelled.")
    
    reasons = []
    if bool_passing:
        reasons.append("passing")
    if bool_neck:
        reasons.append("neck")
    if bool_bud:
        reasons.append("bud")
    if bool_dim:
        reasons.append("dim")
    if bool_oof:
        reasons.append("out_of_focus")
    if bool_overlap:
        reasons.append("spot_overlap")
    if bool_bigcluster:
        reasons.append("big_cluster")
    if bool_startend:
        reasons.append("start_end_not_shown")
    if bool_short:
        reasons.append("short")
    if bool_center:
        reasons.append("center_cell")
    if bool_nuf2:
        reasons.append("nuf2")
    if other != "":
        reasons.append(str(other))

    print "Reasons for discarding:", reasons, "\n"
    return reasons

########################################################

class NonBlockingCurationDialog(JDialog):
    def __init__(self, latch, location = (100,100), title="Manual annotation"):
        # Create a non-modal dialog so that other windows remain accessible.
        JDialog.__init__(self)
        self.setTitle(title)
        self.setModal(False)
        self.latch = latch
        
        # Use a vertical BoxLayout for stacking components
        self.getContentPane().setLayout(BoxLayout(self.getContentPane(), BoxLayout.Y_AXIS))
        
        # Buttons
        self.radio_bona_fide = JRadioButton("Bona fide")
        self.radio_ambiguous = JRadioButton("Ambiguous")
        self.radio_unclear = JRadioButton("Unclear")

        self.group = ButtonGroup()
        for rb in [self.radio_bona_fide, self.radio_ambiguous, self.radio_unclear]:
            self.group.add(rb)
            panel = JPanel(FlowLayout(FlowLayout.LEFT))
            panel.add(rb)
            self.getContentPane().add(panel)
            
        # Stop button
        panelStop = JPanel(FlowLayout(FlowLayout.CENTER))
        self.buttonStop = JButton("Stop classifying")
        self.buttonStop.addActionListener(lambda e: self.onStop())
        panelStop.add(self.buttonStop)
        self.getContentPane().add(panelStop)
        # OK button
        panelOk = JPanel(FlowLayout(FlowLayout.CENTER))
        self.buttonOk = JButton("OK")
        self.buttonOk.addActionListener(lambda e: self.onOk())
        panelOk.add(self.buttonOk)
        self.getContentPane().add(panelOk)
        
        # Set preferred size and pack the dialog
        self.setPreferredSize(Dimension(200, 200))
        self.pack()
        self.setLocation(location[0], location[1])  # Position the dialog at a specific location
        
        # This variable will hold the user selection result as a tuple: (label, stop)
        self.selectionResult = None
           
    def onOk(self):
        
        if self.radio_bona_fide.isSelected():
            label = "bona_fide"
        elif self.radio_ambiguous.isSelected():
            label = "ambiguous"
        elif self.radio_unclear.isSelected():
            label = "unclear"
        else:
            label = None  # No selection

        self.selectionResult = (label, False)
        self.setVisible(False)
        self.latch.countDown()
        
    def onStop(self):
        self.selectionResult = (None, True)
        self.setVisible(False)
        self.latch.countDown()

    def getSelection(self):
        return self.selectionResult


# Example of how to use this custom dialog:
def show_gui_manual_curation(info_track, writer, filename_csv_all, location_gui = (100,100), show_reason_discarded = True):
    """
    Launch the manual curation GUI, retrieve the user's selection, and write to CSV.
    """
    latch = CountDownLatch(1)
    dlg = NonBlockingCurationDialog(latch, location = location_gui, title="Manual annotation")
    dlg.setVisible(True)  # This call blocks until the dialog is hidden
    # Wait until the user clicks OK or Stop
    latch.await()  # This will block the main thread until latch.countDown() is called

    # Retrieve selections as booleans
    label, stop_bool = dlg.getSelection()
    
    if stop_bool:
        IJ.run("Close All")
        IJ.selectWindow(filename_csv_all); IJ.run("Close")
        exit()
        
    if label is None:
        print("**********************")
        print("No selection was done!")
        print("**********************\n")
        writer.writerow(["not_labeled"] + info_track + [""])
    else:
        print("Labeled as \"%s\"\n"%(label))
    
    if label == "bona_fide":
        writer.writerow(["bona_fide"] + info_track + [""])
    elif label == "ambiguous":
        if show_reason_discarded:
            reasons_discarded = gui_reason_discarded()
            reasons_discarded = "|".join(reasons_discarded)
            writer.writerow(["ambiguous"] + info_track + [reasons_discarded])
        else:
            writer.writerow(["ambiguous"] + info_track + [""])
    elif label == "unclear":
        writer.writerow(["unclear"] + info_track + [""])

########################################################
########################################################
########################################################

def load_trackmate_csv(path_csv, filename_csv_all, relevant_columns):
    """
    Opens TrackMate CSV and checks required columns.
    Returns:
        TM_table
        relevant_columns_idx
        column_names
    """
    path_csv_all = os.path.join(path_csv, filename_csv_all)
    if not os.path.exists(path_csv_all):
        raise IOError("CSV file with non-curated tracks not found: " + path_csv_all)

    IJ.open(path_csv_all)
    TM_table = ResultsTable.getResultsTable(filename_csv_all)

    if TM_table is None or TM_table.size() == 0:
        raise ValueError("No data found in CSV file: " + filename_csv_all)

    relevant_columns_idx = {}
    for column in relevant_columns:
        idx = TM_table.getColumnIndex(column)
        if idx == -1:
            raise RuntimeError("CSV missing required column: %s" % column)
        relevant_columns_idx[column] = idx

    column_names = TM_table.getColumnHeadings().split("\t")
    


    return TM_table, relevant_columns_idx, column_names

########################################################

def get_file_ids(TM_table, relevant_columns_idx):
    """
    Returns the set of FILE_IDs present in the CSV.
    """
    column_FILE_ID = [int(x) for x in TM_table.getColumnAsDoubles(relevant_columns_idx["FILE_ID"])]
    file_ids = sorted(set(column_FILE_ID))

    return file_ids

########################################################

def load_already_curated_tracks(curated_csv_path, relevant_columns):
    """
    Load tracks already curated in the curated CSV file.
    """
    if not os.path.exists(curated_csv_path):
        return []
        
    already_curated = []
    
    with open(curated_csv_path, "rb") as f:
        reader = csv.reader(f)
        
        header = next(reader)
        relevant_columns_curated_idx = {}
        for column in relevant_columns:
            relevant_columns_curated_idx[column] = header.index(column)

        for row in reader:
   
            relevant_elements = {}
            for column_name, column_idx in relevant_columns_curated_idx.items():
                try:
                    relevant_elements[column_name] = int(float(row[column_idx]))
                except:
                    relevant_elements[column_name] = str(row[column_idx])

            already_curated.append(relevant_elements)

    return already_curated
    
    
def load_already_curated(path):
    """
    Load the (FILE_ID, TRACK_ID) pairs already present in the curated CSV.

    Returns:
    
        already_curated (set of tuples): {(FILE_ID, TRACK_ID), ...}
    """
    already_curated = set()

    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return already_curated

    with open(path, "rb") as f:
        for row in csv.DictReader(f):
            try:
                file_id  = int(float(row["FILE_ID"]))
                track_id = int(float(row["TRACK_ID"]))
                already_curated.add((file_id, track_id))
            except (KeyError, ValueError):
                continue  # skip malformed rows

    return already_curated
    
########################################################

def group_uncurated_tracks(TM_table, already_curated):
    """
    Build a mapping of FILE_ID → [row indices] for every track
    that has not yet been curated.

    Parameters
    ----------
    TM_table       : ResultsTable.
    already_curated: set of (FILE_ID, TRACK_ID) tuples.

    Returns
    -------
    dict  {file_id (int): [row_index, ...]}
    Only FILE_IDs that have at least one uncurated track appear as keys,
    so iterating over this dict automatically skips files with no work to do.
    """
    tracks_by_file = defaultdict(list)

    for i in range(TM_table.size()):
        file_id  = int(TM_table.getStringValue("FILE_ID",  i))
        track_id = int(TM_table.getStringValue("TRACK_ID", i))

        if (file_id, track_id) not in already_curated:
            tracks_by_file[file_id].append(i)

    return tracks_by_file
    
########################################################

def open_fov_image(fov_path):
    """
    Opens a FOV image using BioFormats with fallback to IJ.openImage.
    """
    
    # First attempt: Bio-Formats
    try:
        whole_FOV = Opener.openUsingBioFormats(fov_path)

        if whole_FOV is not None:
            return whole_FOV
        else:
            print "Bio-Formats returned None for '%s'" % fov_path

    except Exception as e:
        print "Bio-Formats failed to open '%s': %s" % (fov_path, e)
    
    # Fallback: IJ.openImage
    print "Attempting fallback using IJ.openImage()..."
    whole_FOV = IJ.openImage(fov_path)
    
    if whole_FOV is None:
        raise IOError("Failed to open image with both Bio-Formats and IJ.openImage(): " + fov_path)

    return whole_FOV
    
########################################################

def write_csv_header(writer, column_names, show_reason_discarded):
    """
    Writes the header row to the CSV, if needed.
    """
    if show_reason_discarded:
        writer.writerow(["MANUAL_ID"] + column_names + ["REASON_DISCARDED"])
    else:
        writer.writerow(["MANUAL_ID"] + column_names)
       
########################################################

def process_track(whole_FOV, crop_size, track_x, track_y, track_start, track_stop, 
    extra_frames, zoom, info_track, writer, show_profile, location_gui_x, location_gui_y, show_reason_discarded, filename_csv_all):

    crop, start_frame_crop, stop_frame_crop = crop_from_coordinates(
        whole_FOV, 
        crop_size, 
        x_center = track_x, 
        y_center = track_y, 
        track_start = track_start, 
        track_stop = track_stop, 
        extra_frames = extra_frames,
    )
    
    gaussian_crop = display_gaussian(
        raw = crop, 
        track_start = track_start, 
        track_stop = track_stop, 
        crop_start = start_frame_crop,
        zoom = zoom, 
        gaussian_blur_parameters = (0.5, 0.5, 1.5), 
        ROI_diameter = 10,
    )

    if show_profile:
        dog_sigma1 = 1.24
        dog_sigma2 = 1.76
        display_intensity_profile(gaussian_crop, dog_sigma1, dog_sigma2)
        IJ.run("Tile")

    show_gui_manual_curation(
        info_track = info_track, 
        writer = writer, 
        filename_csv_all = filename_csv_all,
        location_gui = (location_gui_x, location_gui_y), 
        show_reason_discarded = show_reason_discarded,
    )
                
    IJ.run("Close All")
    IJ.run("Collect Garbage")
        
########################################################

def run_everything(params):
    """
    Main workflow:
    - Opens the CSV with non-curated tracks.
    - Checks required columns.
    - Iterates through FOVs corresponding to tracks.
    - Opens each FOV, processes tracks, and displays GUI for manual curation.
    - Saves curated track info to a new CSV.
    """
    
    experiment = params["experiment"]
    path_movies = params["path_movies"]
    path_csv = params["path_csv"]
    filename_csv_all = params["filename_csv_all"]
    filename_csv_curated = params["filename_csv_curated"]
    show_reason_discarded = params["show_reason_discarded"]
    show_profile = params["show_profile"]
    crop_size = params["crop_size"]
    zoom = params["zoom"]
    extra_frames = params["extra_frames"]
    extra_name_fov = params["extra_name_fov"]
    location_gui_x = params["location_gui_x"]
    location_gui_y = params["location_gui_y"]
    
    relevant_columns = ["EXPERIMENT", "FILE_ID", "TRACK_ID", "TRACK_START", "TRACK_STOP"]
    
    # Load TrackMate CSV
    TM_table, relevant_columns_idx, column_names = load_trackmate_csv(path_csv, filename_csv_all, relevant_columns)
    
    num_tracks = TM_table.size()
    num_columns = len(column_names)
    
    file_ids = get_file_ids(TM_table, relevant_columns_idx)

    print "\n*************************************"
    print "\tNumber of files: %d" % (len(file_ids))
    print "\tFile IDs:", file_ids
    print "*************************************"
    
    # Curated CSV
    curated_csv_path = os.path.join(path_csv, filename_csv_curated)
    already_curated = load_already_curated(curated_csv_path)
    tracks_by_file = group_uncurated_tracks(TM_table, already_curated)

    print "\n*************************************"
    print "\tTotal tracks     : %d" % num_tracks
    print "\tAlready curated  : %d" % len(already_curated)
    print "\tRemaining tracks : %d" % sum(len(v) for v in tracks_by_file.values())
    print "\tFiles to open    : %d" % len(tracks_by_file)
    print "*************************************\n
    
    if not tracks_by_file:
        print "All tracks have already been curated. Nothing to do."
        return

    # Prepare output CSV (append mode; write header only when creating the file)
    header_already_written = os.path.exists(curated_csv_path) and os.path.getsize(curated_csv_path) > 0
    
    with open(curated_csv_path, "ab") as curated: 
        writer = csv.writer(curated)
        
        if not header_already_written:
            write_csv_header(writer, column_names, show_reason_discarded)
    
        # Iterate only over files that have uncurated tracks
        curated_in_session = 0
        
        for file_id, track_indices in sorted(tracks_by_file.items()):
            print "\nProcessing FILE_ID %d (%d uncurated tracks)" % (file_id, len(track_indices))
            filename_wholeFOV = "%s_%s_%d.tif" % (experiment, extra_name_fov, file_id)
            print "FOV:", filename_wholeFOV
            
            fov_path = os.path.join(path_movies, filename_wholeFOV)
            if not os.path.exists(fov_path):
                print("Warning: FOV file not found: " + fov_path)
                continue

            whole_FOV = open_fov_image(fov_path)
            width, height, frames = whole_FOV.getWidth(), whole_FOV.getHeight(), whole_FOV.getNFrames()
            print "\tImage dimensions: width=%d, height=%d, frames=%d\n" % (width, height, frames)
            
            
            # Process only the pre-filtered row indices for this file
            for i in track_indices:
                
                info_track = [TM_table.getValueAsDouble(col, i) for col in range(num_columns)]
                info_track[TM_table.getColumnIndex("EXPERIMENT")] = TM_table.getStringValue("EXPERIMENT", i)
                     
                track_start = int(TM_table.getStringValue("TRACK_START", i))
                track_stop = int(TM_table.getStringValue("TRACK_STOP", i))
                track_x = float(TM_table.getStringValue("TRACK_X_LOCATION", i))
                track_y = float(TM_table.getStringValue("TRACK_Y_LOCATION", i))
                track_id = int(TM_table.getStringValue("TRACK_ID", i))
                
                print "Track %d/%d: start=%d | stop=%d | x=%.2f | y=%.2f" % (len(already_curated)+curated_in_session+1, num_tracks, track_start, track_stop, track_x, track_y)

                process_track(
                    whole_FOV = whole_FOV,
                    crop_size = crop_size,
                    track_x = track_x,
                    track_y = track_y,
                    track_start = track_start,
                    track_stop = track_stop,
                    extra_frames = extra_frames,
                    zoom = zoom,
                    info_track = info_track,
                    writer = writer,
                    show_profile = show_profile,
                    location_gui_x = location_gui_x, 
                    location_gui_y = location_gui_y, 
                    show_reason_discarded = show_reason_discarded,
                    filename_csv_all =filename_csv_all,
                )
                curated_in_session += 1
            whole_FOV.close()
            
    print "\nAnnotation has finished"

########################################################

def show_dialog_initial():
    """
    Displays a GUI for initial user input with default values that persist between sessions.

    Returns
    -------
    dict
        Dictionary containing all user-selected parameters.
    """
    
    # Default values
    DEFAULTS = {
        "crop_size": 40,
        "zoom": 800,
        "extra_frames": 15,
        "extra_name_fov": "prepro_C1",
        "extra_name_curated": "_curated",
        "reason_discarded": True,
        "show_profile": True,
        "location_gui_x": 100,
        "location_gui_y": 100
    }

    # Load preferences
    default_experiment          = prefs.get(None, "experiment", "")
    default_root                = prefs.get(None, "root_folder", IJ.getDirectory("home"))
    default_csv                 = prefs.get(None, "csv_file", "")
    
    default_crop_size           = prefs.getFloat(None, "crop_size", DEFAULTS["crop_size"])
    default_zoom                = prefs.getFloat(None, "zoom", DEFAULTS["zoom"])
    default_extra_frames        = prefs.getInt(None, "extra_frames", DEFAULTS["extra_frames"])

    default_extra_name_fov      = prefs.get(None, "extra_name_fov", DEFAULTS["extra_name_fov"])
    default_extra_name_curated  = prefs.get(None, "extra_name_curated", DEFAULTS["extra_name_curated"])

    default_reason_discarded    = bool(prefs.getInt(None, "reason_discarded", DEFAULTS["reason_discarded"]))
    default_show_profile        = bool(prefs.getInt(None, "show_profile", DEFAULTS["show_profile"]))
    
    default_location_gui_x      = prefs.getInt(None, "location_gui_x", DEFAULTS["location_gui_x"])
    default_location_gui_y      = prefs.getInt(None, "location_gui_y", DEFAULTS["location_gui_y"])
    
    # Create dialog
    dialog = GenericDialogPlus("Track annotation setup")
    
    dialog.addStringField("Experiment name:", default_experiment)
    dialog.addToSameRow()
    dialog.addMessage("Video names must follow:\n   ExperimentName_ExtraNameFOV_FileID.tif")
    
    dialog.addDirectoryField("Root folder:", default_root, 40)
    dialog.addFileField("CSV file (non-curated):", default_csv, 40)
    
    dialog.addCheckbox("Enable reason selection for discarded tracks", default_reason_discarded)
    dialog.addCheckbox("Enable display of track intensity profile", default_show_profile)
    
    dialog.addMessage("Display Parameters:")
    
    dialog.addNumericField("Crop size (for display):", default_crop_size, 0)
    dialog.addToSameRow()
    dialog.addMessage("(Crop size around the track center)")
    
    dialog.addNumericField("Zoom (for display):", default_zoom, 0)
    dialog.addToSameRow()
    dialog.addMessage("(Zoom factor for combined image display)")
    
    dialog.addNumericField("Extra frames (for display):", default_extra_frames, 0)
    dialog.addToSameRow()
    dialog.addMessage("(Frames before/after track)")
    
    # Screen info
    screen_size = Toolkit.getDefaultToolkit().getScreenSize()
    screen_width = int(screen_size.getWidth())
    screen_height = int(screen_size.getHeight())
    
    dialog.addMessage("Screen size: {} x {}".format(screen_width, screen_height))
    
    dialog.addNumericField("Location GUI (x):", default_location_gui_x, 0)
    dialog.addNumericField("Location GUI (y):", default_location_gui_y, 0)
    
    dialog.addMessage("Naming Parameters:")
    
    dialog.addStringField("Extra name FOV", default_extra_name_fov)
    dialog.addToSameRow()
    dialog.addMessage("(Text after experiment name in video names)")
    
    dialog.addStringField("Extra name curated CSV", default_extra_name_curated)
    dialog.addToSameRow()
    dialog.addMessage("(Text added to curated CSV filename)")

    dialog.showDialog()
    
    if not dialog.wasOKed():
        raise Exception("Initial GUI was cancelled.")

    # Retrieve user inputs 
    experiment       = dialog.getNextString()
    root_folder      = dialog.getNextString()
    csv_file         = dialog.getNextString()
    
    show_reason_discarded   = dialog.getNextBoolean()
    show_profile            = dialog.getNextBoolean()
    
    crop_size        = dialog.getNextNumber()
    zoom             = dialog.getNextNumber()
    extra_frames     = int(dialog.getNextNumber())
    
    location_gui_x   = int(dialog.getNextNumber())
    location_gui_y   = int(dialog.getNextNumber())
    
    extra_name_fov          = dialog.getNextString()
    extra_name_curated      = dialog.getNextString()
    
    # Validate GUI position
    if location_gui_x < 0 or location_gui_x > screen_width:
        print("X coordinate out of screen bounds. Resetting to default.")
        location_gui_x = default_location_gui_x
    
    if location_gui_y < 0 or location_gui_y > screen_height:
        print("Y coordinate out of screen bounds. Resetting to default.")
        location_gui_y = default_location_gui_y
        
    # Save preferences
    prefs_dict = {
        "experiment": experiment,
        "root_folder": root_folder,
        "csv_file": csv_file,
        "reason_discarded": show_reason_discarded,
        "show_profile": show_profile,
        "crop_size": crop_size,
        "zoom": zoom,
        "extra_frames": extra_frames,
        "extra_name_fov": extra_name_fov,
        "extra_name_curated": extra_name_curated,
        "location_gui_x": location_gui_x,
        "location_gui_y": location_gui_y
    }
    
    for key, value in prefs_dict.items():
        prefs.put(None, key, value)
        
    # ------------------------------------------------------------------
    # Process paths
    # ------------------------------------------------------------------
    root_folder = os.path.normpath(root_folder)
    path_movies = os.path.join(root_folder, "3_Preprocessed")
    
    filename_csv_all = os.path.basename(csv_file)
    path_csv = os.path.dirname(csv_file)

    base, ext = os.path.splitext(filename_csv_all)
    filename_csv_curated = base + extra_name_curated + ext
    
    return {
        "experiment": experiment,
        "root_folder": root_folder,
        "path_movies": path_movies,
        "path_csv": path_csv,
        "filename_csv_all": filename_csv_all,
        "filename_csv_curated": filename_csv_curated,
        "show_reason_discarded": show_reason_discarded,
        "show_profile": show_profile,
        "crop_size": crop_size,
        "zoom": zoom,
        "extra_frames": extra_frames,
        "extra_name_fov": extra_name_fov,
        "location_gui_x": location_gui_x,
        "location_gui_y": location_gui_y
    }


########################################################
###################### USER INPUT ######################
########################################################

IJ.run("Collect Garbage") # clean memory

params = show_dialog_initial()

########################################################
#################### RUN EVERYTHING ####################
########################################################

print "\n*******************************"
print "Root folder:", params["root_folder"]
print "CSV to curate:", params["filename_csv_all"]
print "*******************************"

run_everything(params)
