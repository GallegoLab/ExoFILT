"""
Script:     Automated tracking of exocytic events using TrackMate
Version:    3.0.1
Author:     Eric Kramer i Rosado

Description
-----------
    Perform automated detection and tracking of exocytic events in single-color live-cell imaging data.
    The script exports CSV tables containing spot-level and track-level features for downstream analysis.
    
Pipeline
-----------
1) Load movie
2) Preprocess image (Gaussian smoothing + DoG generation)
3) Detect and track particles using TrackMate
4) Compute additional quality metrics
5) Export results to CSV files
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Python standard library
import csv
import math
import os

# ImageJ
from ij import IJ
from ij.io import Opener
from ij.measure import Calibration
from ij.gui import OvalRoi
from ij.plugin import ImageCalculator

# TrackMate
from fiji.plugin.trackmate import Model, Settings, TrackMate, SelectionModel, Logger
from fiji.plugin.trackmate.detection import DogDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
from fiji.plugin.trackmate.features import FeatureFilter

# TrackMate visualization
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.gui.wizard.TrackMateWizardSequence as TrackMateWizardSequence
import fiji.plugin.trackmate.gui.GuiUtils as GuiUtils
import fiji.plugin.trackmate.gui.wizard.descriptors.ConfigureViewsDescriptor as ConfigureViewsDescriptor
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO

# GUI
from fiji.util.gui import GenericDialogPlus

#@ PrefService prefs 

# ------------------------------------------------------------------------------
# Run everything through each file
# ------------------------------------------------------------------------------

def run():
    """ 
    Iterate over all files in input_path containing the C1 identifier and run the processing pipeline.
    """
    files = sorted([f for f in os.listdir(input_path) if contain_string_C1 in f])
    
    print "\n*******************************"
    print "Number of files to process:%d" % (len(files))
    print "*******************************"
    
    if not files:
        print "No files matching the criteria found in %s" % input_path
        return
        
    for i, filename in enumerate(files):
        print "\nProcessing file %d/%d: %s" %(i+1, len(files), filename)
        # Check for file extension
        if not filename.endswith(file_extension):
            print "Filename does not have the required extension (%s). File skipped: %s" % (file_extension, filename)
            continue
        
        try:
            process(input_path, output_path, filename)
            
        except Exception as e:
            print "Error processing file '%s': %s" % (filename, e)
            IJ.run("Close All", "")
            continue

# ------------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------------

def preprocess(imp):
    
    """
    Preprocess the input movie prior to tracking.

    Steps
    -----
    1. Set calibration to pixel units.
    2. Validate image dimensions.
    3. Apply 3D Gaussian smoothing.
    4. Generate a Difference-of-Gaussians (DoG) image for quality measurements.

    Returns
    -------
    (imp_processed, imp_DoG)
    """

    print "\tPreprocessing..."
    
    # Set calibration parameters
    newCal = Calibration()
    newCal.pixelWidth = 1
    newCal.pixelHeight = 1
    newCal.frameInterval = 1
    newCal.setXUnit("pixel")
    newCal.setYUnit("pixel")
    newCal.setTimeUnit("unit")
    
    imp.setCalibration(newCal)
    cal = imp.getCalibration()
    
    # Verify image dimensions and calibration
    width = imp.getWidth()
    height = imp.getHeight()
    num_frames = imp.getNFrames()
    
    print"\tImage dimensions: width=%d, height=%d, frames=%d" % (width, height, num_frames)
    
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid image dimensions: width or height is zero or negative!")
    if cal.pixelWidth <= 0 or cal.pixelHeight <= 0:
        raise RuntimeError("Invalid calibration: pixel dimensions must be positive!")
    if num_frames < 1:
        raise RuntimeError("Image contains no frames.")
    
    # Apply a 3D Gaussian blur to reduce noise
    IJ.run(imp,"Gaussian Blur 3D...", "x=0.5 y=0.5 z=2")                        
    imp_processed = imp
    
    # Create Difference-of-Gaussian (DoG) image
    sigma1 = 1.24
    sigma2 = 1.76
        
    imp_gauss_small = imp_processed.duplicate()
    IJ.run(imp_gauss_small, "32-bit", "")
    imp_gauss_large = imp_gauss_small.duplicate()
        
    IJ.run(imp_gauss_small, "Accurate Gaussian Blur", "sigma=%f stack" % sigma1)
    IJ.run(imp_gauss_large, "Accurate Gaussian Blur", "sigma=%f stack" % sigma2)
        
    # Subtract one smoothed image from the other
    imp_DoG = ImageCalculator.run(imp_gauss_small, imp_gauss_large, "Subtract create stack") 
    imp_DoG.setTitle("DoG")
    imp_gauss_small.close()
    imp_gauss_large.close()
    
    return (imp_processed,imp_DoG)

# ------------------------------------------------------------------------------
# TrackMate processing
# ------------------------------------------------------------------------------

def process(input_path, output_path, filename):
    """
    Process one file:
      - Open the image and extract info from the filename.
      - Preprocess the image.
      - Run TrackMate tracking and calculate additional quality features.
      - Write CSV output files (for spots and tracks).
      - Optionally display the results.
    """
    # Parse filename
    filename_noext = os.path.splitext(filename)[0] # Remove file extension (.tif)
    parts = filename_noext.split("_")

    try:
        file_id = int(parts[-1])
    except Exception as e:
        raise RuntimeError("Error parsing file_id from filename: %s. Expected format: *C1_<id>.tif" % filename)

    channel = parts[-2]
    experiment_name = "_".join(parts[:-3]) # Get the name that identifies the experiment: filename without "prepro", channel and file_id
    experiment_channel_id = "%s_%s_%d" % (experiment_name, channel, file_id)

    # Open image
    image_path = os.path.join(input_path, filename)
    imp = None
    try:
        imp = Opener.openUsingBioFormats(image_path)
        if imp is None:
            raise RuntimeError("Bio-Formats returned None")
    except Exception as e:
        print "Bio-Formats failed to open '%s': %s" % (filename, e)
        print "Attempting fallback using IJ.openImage()..."
    
    # Fallback: ImageJ opener
    if imp is None:
        imp = IJ.openImage(image_path)

    # Final check
    if imp is None:
        raise RuntimeError("Failed to open image with both Bio-Formats and IJ.openImage(): %s" % filename)

    # Preprocess image
    imp,imp_DoG = preprocess(imp)
    
    # Ensure that the DoG image is available
    if imp_DoG is None:
        raise RuntimeError("DoG image not available")

    cal = imp.getCalibration()
    width = imp.getWidth()
    height = imp.getHeight()
    num_frames = imp.getNFrames()
    
    # Start the tracking
    print "\tRunning TrackMate..."
    model = Model()
    
    #Read the image calibration
    model.setPhysicalUnits(cal.getUnit(), cal.getTimeUnit())
    # Send all messages to ImageJ log window
    model.setLogger(Logger.IJ_LOGGER)

    settings = Settings(imp)

    # Configure detector
    settings.detectorFactory = DogDetectorFactory()
    settings.detectorSettings = {
        'RADIUS' : 1.5,                             # The radius of the object to detect, in physical units
        'TARGET_CHANNEL' : 1,
        'THRESHOLD' : INIT_Q,                       # Threshold value on quality below which detected spots are discarded
        'DO_SUBPIXEL_LOCALIZATION' : True,          # If True the spot position will be refined with sub-pixel accuracy (quadratic fitting scheme)
        'DO_MEDIAN_FILTERING' : False,              # If True the input will be processed by a 2D 3x3 median before detection
    }
    
    # Configure spot filters
    settings.addSpotFilter(FeatureFilter('FRAME', 10, True)) # Exclude first 10 frames
    settings.addSpotFilter(FeatureFilter('FRAME', num_frames-10, False)) # Exclude last 10 frames
    settings.addSpotFilter(FeatureFilter('SNR_CH1', 0.2, True))
    settings.addSpotFilter(FeatureFilter('MIN_INTENSITY_CH1', 5, True)) # Exclude low-intensity spots
    settings.addSpotFilter(FeatureFilter('POSITION_X', 10, True))
    settings.addSpotFilter(FeatureFilter('POSITION_X', width-10, False))
    settings.addSpotFilter(FeatureFilter('POSITION_Y', 10, True))
    settings.addSpotFilter(FeatureFilter('POSITION_Y', height-10, False))
    
    # Configure tracker
    settings.trackerFactory = SparseLAPTrackerFactory()
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
    settings.trackerSettings['LINKING_MAX_DISTANCE'] = LINKING_MAX_DISTANCE # The max distance between two consecutive spots, in physical units, allowed for creating links
    settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = ALLOW_TRACK_SPLITTING # If True the tracker will perform tracklets or segments splitting, that is: have one tracklet ending linking to two or more tracklet beginnings
    settings.trackerSettings['SPLITTING_MAX_DISTANCE'] = SPLITTING_MAX_DISTANCE # Track splitting max spatial distance
    settings.trackerSettings['ALLOW_TRACK_MERGING'] = ALLOW_TRACK_MERGING # If True then the tracker will perform tracklets or segments merging, that is: have two or more tracklet endings linking to one tracklet beginning
    settings.trackerSettings['MERGING_MAX_DISTANCE'] = MERGING_MAX_DISTANCE # Track merging max spatial distance
    settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = GAP_CLOSING_MAX_DISTANCE # max distance between two spots, in physical units, allowed for creating links over missing detections
    settings.trackerSettings['MAX_FRAME_GAP'] = MAX_FRAME_GAP # max difference in time-points between two spots to allow for linking --> a value of 2 bridges over one missed detection in one frame

    # Add all available analyzers
    settings.addAllAnalyzers()
    # Configure track filters
    settings.addTrackFilter(FeatureFilter('TRACK_DURATION', 1/time_interval, True))
    settings.addTrackFilter(FeatureFilter('TRACK_MEAN_QUALITY', MEAN_Q, True) )
    
    # Instantiate TrackMate plugin
    trackmate = TrackMate(model, settings)
    # Check input and process
    ok = trackmate.checkInput()
    if not ok:
        raise RuntimeError("TrackMate input error: " + str(trackmate.getErrorMessage()))
    ok = trackmate.process()
    if not ok:
        raise RuntimeError("TrackMate processing error: " + str(trackmate.getErrorMessage()))
    
    # Feature model: stores edge and track features
    fm = model.getFeatureModel()

    # Calculate additional Quality Features per track
    for id in model.getTrackModel().trackIDs(True):
        try:
            # Retrieve overall track features
            track_start = fm.getTrackFeature(id, 'TRACK_START') 
            track_stop = fm.getTrackFeature(id, 'TRACK_STOP') 
            x_mean = fm.getTrackFeature(id, 'TRACK_X_LOCATION') 
            y_mean = fm.getTrackFeature(id, 'TRACK_Y_LOCATION')
        except Exception as e:
            print "Warning: Unable to retrieve some track features for track %s: %s" % (id, e)
            continue
        
        # Get the list of spots that belong to this track
        track = model.getTrackModel().trackSpots(id)
        
        # Initialize variables to calculate quality
        q_track_in_mean  = 0
        q_track_env_mean = 0

        q_in_aft  = 0
        q_in_bef  = 0
        
        # Calculate Q_IN and Q_ENV for each spot in the track
        for spot in track:
            x = spot.getFeature('POSITION_X')
            y = spot.getFeature('POSITION_Y')
            frame = spot.getFeature('FRAME')
            radius = spot.getFeature('RADIUS')
            
            # Q_IN: intensity measured directly within ROI defined by the spot radius
            imp_DoG.setSlice(int(frame + 1))
            imp_DoG.setRoi(OvalRoi(x-radius+0.5, y-radius+0.5, 2*radius, 2*radius))
            stats_r = imp_DoG.getStatistics()
            q_in_mean = stats_r.mean
            q_track_in_mean += q_in_mean
            
            # Q_ENV: environmental intensity computed from two differently sized ROIs
            # Second ROI: slightly larger region (radius + 0.5)
            r_adjusted = radius + 0.5
            imp_DoG.setSlice(int(frame + 1))
            imp_DoG.setRoi(OvalRoi(x-r_adjusted+0.5, y-r_adjusted+0.5, 2*r_adjusted, 2*r_adjusted))          
            stats_r_adj = imp_DoG.getStatistics()
            a_r_adj = stats_r_adj.area
            q_r_adj_mean = stats_r_adj.mean
            
            # Third ROI: a region with twice the original radius
            r_double = 2 * radius
            imp_DoG.setSlice(int(frame + 1))
            imp_DoG.setRoi(OvalRoi(x-r_double+0.5, y-r_double+0.5, 2*r_double, 2*r_double))
            stats_r_double = imp_DoG.getStatistics()
            a_r_double = stats_r_double.area
            q_r_double_mean = stats_r_double.mean

            # Compute environmental quality mean (Q_ENV) as a difference quotient
            if (a_r_double - a_r_adj) != 0:
                q_env_mean = (q_r_double_mean*a_r_double - q_r_adj_mean*a_r_adj)/(a_r_double-a_r_adj)
            else:
                q_env_mean = 0
            q_track_env_mean += q_env_mean
            
        # Compute quality before and after start/end, leaving a gap of `time_window_bef_aft`
        
        # Measure intensity in the frames before track_start, leaving a gap
        for i in range(time_window_bef_aft):
            imp_DoG.setSlice(int(track_start - time_window_bef_aft - i + 1))
            imp_DoG.setRoi(OvalRoi(x_mean-radius+0.5, y_mean-radius+0.5, 2*radius, 2*radius))
            stats_r = imp_DoG.getStatistics()
            q_in_bef += stats_r.mean
            
        # Measure intensity in the frames after track_stop, leaving a gap
        for i in range(time_window_bef_aft):
            imp_DoG.setSlice(int(track_stop + time_window_bef_aft + i + 1))  # +1 because setSlice is 1-based
            imp_DoG.setRoi(OvalRoi(x_mean-radius+0.5, y_mean-radius+0.5, 2*radius, 2*radius))
            stats_r = imp_DoG.getStatistics()
            q_in_aft += stats_r.mean
            
        # Compute averages
        q_in_aft  = q_in_aft/time_window_bef_aft
        q_in_bef  = q_in_bef/time_window_bef_aft
        q_track_in_mean = q_track_in_mean/(track_stop-track_start+1)
        q_track_env_mean = q_track_env_mean/(track_stop-track_start+1)

        fm.putTrackFeature(id,"TRACK_MEAN_Q_IN",q_track_in_mean)
        fm.putTrackFeature(id,"TRACK_MEAN_Q_ENV",q_track_env_mean)
        fm.putTrackFeature(id,"TRACK_MEAN_Q_IN_AFTER",q_in_aft)
        fm.putTrackFeature(id,"TRACK_MEAN_Q_IN_BEFORE",q_in_bef)
  
    #----------------
    # Writing CSV results 
    #----------------
    
    # Ensure output subdirectories exist
    path_allspots = os.path.join(output_path, "individual_movies_all_spots")
    path_spotsmodified = os.path.join(output_path, "individual_movies_spots_modified")
    path_tracks = os.path.join(output_path, "individual_movies_tracks")
    
    for path in [path_allspots, path_spotsmodified, path_tracks]:
        if not os.path.exists(path):
            os.mkdir(path)
    
    filename_all_spots_csv = os.path.join(path_allspots, experiment_channel_id + "_allspots.csv")
    filename_spots_modified = os.path.join(path_spotsmodified, experiment_channel_id + "_spotsmodified.csv")
    filename_tracks = os.path.join(path_tracks, experiment_channel_id + "_tracks.csv")
    
    spots = model.getSpots().iterator(False)
    
    # Write CSV for all spots
    with open(filename_all_spots_csv, "wb") as spotfile:

        writer = csv.writer(spotfile)
        writer.writerow(["QUALITY","POSITION_X","POSITION_Y","FRAME","MEAN_INTENSITY_CH1"])

        for spot in spots:
            writer.writerow([
                spot.getFeature("QUALITY"),
                spot.getFeature("POSITION_X"),
                spot.getFeature("POSITION_Y"),
                spot.getFeature("FRAME"),
                spot.getFeature("MEAN_INTENSITY_CH1")
            ])
    
    # Write CSV for tracks and modified spots
    with open(filename_tracks, "wb") as trackfile:
        writer1 = csv.writer(trackfile)
        writer1.writerow([
            "EXPERIMENT",
            "FILE_ID",
            "TRACK_ID",
            "TRACK_DURATION",
            "TRACK_START",
            "TRACK_STOP",
            "TRACK_X_LOCATION",
            "TRACK_Y_LOCATION",
            "TRACK_DISPLACEMENT",
            "MAX_DISTANCE_TRAVELED",
            "TRACK_MEAN_SPEED",
            "TRACK_MEDIAN_SPEED",
            "TRACK_MAX_SPEED",
            "TRACK_MIN_SPEED",
            "TRACK_STD_SPEED",
            "TRACK_MEAN_QUALITY",
            "TRACK_MEAN_Q_IN",
            "TRACK_MEAN_Q_ENV",
            "TRACK_MEAN_Q_IN_BEFORE",
            "TRACK_MEAN_Q_IN_AFTER",
            "CONFINEMENT_RATIO"])
                          
        with open(filename_spots_modified, "wb") as spotfile:
            writer2 = csv.writer(spotfile)
            writer2.writerow([
                "ID",
                "TRACK_ID",
                "QUALITY",
                "POSITION_X",
                "POSITION_Y",
                "FRAME",
                "RADIUS",
                "MEAN_INTENSITY_CH1",
                "MEDIAN_INTENSITY_CH1",
                "MIN_INTENSITY_CH1",
                "MAX_INTENSITY_CH1",
                "STD_INTENSITY_CH1",
                "CONTRAST_CH1",
                "SNR_CH1"
            ])
        
            for id in model.getTrackModel().trackIDs(True):
                # Fetch the track feature from the feature model
                track = model.getTrackModel().trackSpots(id)
                            
                writer1.writerow([
                    experiment_name, file_id, fm.getTrackFeature(id, "TRACK_ID"),
                    fm.getTrackFeature(id, "TRACK_DURATION"),fm.getTrackFeature(id, "TRACK_START"),fm.getTrackFeature(id, "TRACK_STOP"),
                    fm.getTrackFeature(id, "TRACK_X_LOCATION"),fm.getTrackFeature(id, "TRACK_Y_LOCATION"), 
                    fm.getTrackFeature(id, "TRACK_DISPLACEMENT"),fm.getTrackFeature(id, "MAX_DISTANCE_TRAVELED"),
                    fm.getTrackFeature(id, "TRACK_MEAN_SPEED"),fm.getTrackFeature(id, "TRACK_MEDIAN_SPEED"),
                    fm.getTrackFeature(id, "TRACK_MAX_SPEED"),fm.getTrackFeature(id, "TRACK_MIN_SPEED"),fm.getTrackFeature(id, "TRACK_STD_SPEED"),
                    fm.getTrackFeature(id, "TRACK_MEAN_QUALITY"),fm.getTrackFeature(id, "TRACK_MEAN_Q_IN"),fm.getTrackFeature(id, "TRACK_MEAN_Q_ENV") , 
                    fm.getTrackFeature(id, "TRACK_MEAN_Q_IN_BEFORE"),fm.getTrackFeature(id, "TRACK_MEAN_Q_IN_AFTER"),
                    fm.getTrackFeature(id, "CONFINEMENT_RATIO")
                ])
                
                for spot in track:
                    writer2.writerow([
                        spot.ID(),fm.getTrackFeature(id, "TRACK_ID"),
                        spot.getFeature("QUALITY"),spot.getFeature("POSITION_X"),spot.getFeature("POSITION_Y"),
                        spot.getFeature("FRAME"),spot.getFeature("RADIUS"),
                        spot.getFeature("MEAN_INTENSITY_CH1"),spot.getFeature("MEDIAN_INTENSITY_CH1"),
                        spot.getFeature("MIN_INTENSITY_CH1"),spot.getFeature("MAX_INTENSITY_CH1"),spot.getFeature("STD_INTENSITY_CH1"),
                        spot.getFeature("CONTRAST_CH1"),spot.getFeature("SNR_CH1")])

    tracks_found = model.getTrackModel().nTracks(True)
    print "\tFinished TrackMate: found %d tracks" %(tracks_found)

    #----------------
    # Display results
    #----------------
    if show_tracks:
        imp.show()
        model.getLogger().log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks.')
        selectionModel = SelectionModel(model)
        ds = DisplaySettingsIO.readUserDefault()
        displayer =  HyperStackDisplayer(model, selectionModel, imp, ds)
        displayer.render()
        displayer.refresh()
        sequence = TrackMateWizardSequence( trackmate, selectionModel, ds)
        guiState = ConfigureViewsDescriptor.KEY
        sequence.setCurrent( guiState )
        frame = sequence.run("TrackMate importing CSV ")
        GuiUtils.positionWindow(frame, imp.getWindow())
        frame.setVisible(True)

        IJ.run("Collect Garbage") # clean memory
    else:
        IJ.run("Close All")
        IJ.run("Collect Garbage") # clean memory

# ------------------------------------------------------------------------------
# Combine csv files after processing every file
# ------------------------------------------------------------------------------

def combine_csv_tracks(path_individual, path_all, filename_csv_all):
    
    """ Combine individual track CSV files into one summary CSV. """
    
    print"\n"
    print "***************************"
    print "********* Summary *********"
    print "***************************"
    
    output_csv = os.path.join(path_all, filename_csv_all)
    total_tracks = 0
    first_file = True  # Flag to write the header only once
    
    with open(output_csv, "w") as csv_all:
        writer = csv.writer(csv_all)
        
        for i, filename in enumerate(sorted(os.listdir(path_individual))):
            tracks_i = 0
            path_csv_file_i = os.path.join(path_individual, filename)
            
            with open(path_csv_file_i, "r") as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader)  # Read the header row
                # Write the header to the output file if this is the first file
                if first_file:
                    writer.writerow(header)
                    first_file = False
                # Process and write each row
                for row in reader:
                    total_tracks += 1
                    tracks_i += 1
                    writer.writerow(row)

            print "%s \tNumber of tracks: %d" %(filename, tracks_i)
            
    print "\nA file with all tracks (N=%d) has been created." %(total_tracks)

# ------------------------------------------------------------------------------
# GUI for user input
# ------------------------------------------------------------------------------

# Create a dialog for the user to enter parameters
basicGui = GenericDialogPlus("Tracking Parameters")
basicGui.addDirectoryField("Root folder:", prefs.get(None, "root_folder", IJ.getDirectory("home")))
basicGui.addNumericField("Time interval:",  prefs.getFloat(None, "time_interval", 0.120), 3)
basicGui.addNumericField("INIT_Q:", prefs.getFloat(None, "INIT_Q", 3.0), 2)
basicGui.addToSameRow()
basicGui.addMessage("Lower quality threshold for spot detection")
basicGui.addNumericField("MEAN_Q:", prefs.getFloat(None, "MEAN_Q", 3.5), 2)
basicGui.addToSameRow()
basicGui.addMessage("Lower mean quality threshold for track")
basicGui.addCheckbox("Show advanced options", False)
basicGui.showDialog()
if basicGui.wasCanceled():
    print "User canceled dialog"
    exit()
    
root_folder    = basicGui.getNextString()
time_interval  = round(basicGui.getNextNumber(), 3)
INIT_Q         = round(basicGui.getNextNumber(), 2)
MEAN_Q         = round(basicGui.getNextNumber(), 2)
showAdvanced   = basicGui.getNextBoolean()

# Save in memory using PrefService
prefs.put(None, "root_folder", root_folder)
prefs.put(None, "time_interval", time_interval)
prefs.put(None, "INIT_Q", INIT_Q)
prefs.put(None, "MEAN_Q", MEAN_Q)

# Create a second dialog for advanced parameters
if showAdvanced:

    advGui = GenericDialogPlus("Advanced Tracking Parameters")
    advGui.addCheckbox("Show tracks", bool(prefs.getInt(None, "show_tracks", False)))
    advGui.addStringField("File extension:", prefs.get(None, "file_extension", "tif")) # Extension of the files used for tracking. Other files will be skipped
    advGui.addStringField("String identifying C1 movies:", prefs.get(None, "contain_string_C1", "C1")) # Process only files from channel 1
    advGui.addMessage(" ")
    advGui.addNumericField("LINKING_MAX_DISTANCE:", prefs.getFloat(None, "LINKING_MAX_DISTANCE", 1.5), 2)
    advGui.addNumericField("GAP_CLOSING_MAX_DISTANCE:", prefs.getFloat(None, "GAP_CLOSING_MAX_DISTANCE", 2.0), 2)
    advGui.addNumericField("MAX_FRAME_GAP:", prefs.getInt(None, "MAX_FRAME_GAP", 1), 0)
    advGui.addMessage(" ")
    advGui.addCheckbox("ALLOW_TRACK_SPLITTING", bool(prefs.getInt(None, "ALLOW_TRACK_SPLITTING", False)))
    advGui.addNumericField("SPLITTING_MAX_DISTANCE:", prefs.getFloat(None, "SPLITTING_MAX_DISTANCE", 0.0), 2)
    advGui.addCheckbox("ALLOW_TRACK_MERGING", bool(prefs.getInt(None, "ALLOW_TRACK_MERGING", False)))
    advGui.addNumericField("MERGING_MAX_DISTANCE:", prefs.getFloat(None, "MERGING_MAX_DISTANCE", 0.0), 2)
    advGui.addMessage(" ")
    advGui.addNumericField("Time window for quality before/after:", prefs.getInt(None, "time_window_bef_aft", 14), 0)
    advGui.showDialog()
    
    if advGui.wasCanceled():
        print "User canceled advanced dialog"
        exit()
    else:
        show_tracks                = advGui.getNextBoolean()
        file_extension             = advGui.getNextString()
        contain_string_C1          = advGui.getNextString()
        LINKING_MAX_DISTANCE       = round(advGui.getNextNumber(), 2)
        GAP_CLOSING_MAX_DISTANCE   = round(advGui.getNextNumber(), 2)
        MAX_FRAME_GAP              = int(advGui.getNextNumber())
        ALLOW_TRACK_SPLITTING      = advGui.getNextBoolean()
        SPLITTING_MAX_DISTANCE     = round(advGui.getNextNumber(), 2)
        ALLOW_TRACK_MERGING        = advGui.getNextBoolean()
        MERGING_MAX_DISTANCE       = round(advGui.getNextNumber(), 2)
        time_window_bef_aft        = int(advGui.getNextNumber())

        # Save advanced parameters
        prefs_dict = {
            "show_tracks": show_tracks,
            "file_extension": file_extension,
            "contain_string_C1": contain_string_C1,
            "LINKING_MAX_DISTANCE": LINKING_MAX_DISTANCE,
            "GAP_CLOSING_MAX_DISTANCE": GAP_CLOSING_MAX_DISTANCE,
            "MAX_FRAME_GAP": MAX_FRAME_GAP,
            "ALLOW_TRACK_SPLITTING": ALLOW_TRACK_SPLITTING,
            "SPLITTING_MAX_DISTANCE": SPLITTING_MAX_DISTANCE,
            "ALLOW_TRACK_MERGING": ALLOW_TRACK_MERGING,
            "MERGING_MAX_DISTANCE": MERGING_MAX_DISTANCE,
            "time_window_bef_aft": time_window_bef_aft,
        }
        
        for key, value in prefs_dict.items():
            prefs.put(None, key, value)
        
# Set advanced parameters to defaults or previously stored values
show_tracks                 = bool(prefs.getInt(None, "show_tracks", False))
file_extension              = prefs.get(None, "file_extension", "tif")
contain_string_C1           = prefs.get(None, "contain_string_C1", "C1")
LINKING_MAX_DISTANCE        = round(prefs.getFloat(None, "LINKING_MAX_DISTANCE", 1.5), 2)
GAP_CLOSING_MAX_DISTANCE    = round(prefs.getFloat(None, "GAP_CLOSING_MAX_DISTANCE", 2.0), 2)
MAX_FRAME_GAP               = prefs.getInt(None, "MAX_FRAME_GAP", 1)
ALLOW_TRACK_SPLITTING       = bool(prefs.getInt(None, "ALLOW_TRACK_SPLITTING", False))
SPLITTING_MAX_DISTANCE      = round(prefs.getFloat(None, "SPLITTING_MAX_DISTANCE", 0.0), 2)
ALLOW_TRACK_MERGING         = bool(prefs.getInt(None, "ALLOW_TRACK_MERGING", False))
MERGING_MAX_DISTANCE        = round(prefs.getFloat(None, "MERGING_MAX_DISTANCE", 0.0), 2)
time_window_bef_aft         = prefs.getInt(None, "time_window_bef_aft", 14)


print "\n*******************************"
print "Root folder:", root_folder
print "*******************************\n"

print "Basic parameters:"
print "\tINIT_Q=%s, MEAN_Q=%s, time_interval=%s" % (INIT_Q, MEAN_Q, time_interval)
print "Advanced parameters:"
print "\tLINKING_MAX_DISTANCE=%s, GAP_CLOSING_MAX_DISTANCE=%s, MAX_FRAME_GAP=%d" % (LINKING_MAX_DISTANCE, GAP_CLOSING_MAX_DISTANCE, MAX_FRAME_GAP)
print "\tALLOW_TRACK_SPLITTING=%s, SPLITTING_MAX_DISTANCE=%s, ALLOW_TRACK_MERGING=%s, MERGING_MAX_DISTANCE=%s" % (ALLOW_TRACK_SPLITTING, SPLITTING_MAX_DISTANCE, ALLOW_TRACK_MERGING, MERGING_MAX_DISTANCE)
print "\ttime_window_bef_aft=%d" % (time_window_bef_aft)

input_path = os.path.join(root_folder, "3_Preprocessed")     # Data for tracking should be in a folder named 3_Preprocessed
output_path = os.path.join(root_folder, "4_Tracking")        # Folder where tracking results will be stored

if not os.path.exists(output_path):
    os.mkdir(output_path)

# ------------------------------------------------------------------------------
# Run everything
# ------------------------------------------------------------------------------

IJ.run("Collect Garbage") # Clean memory
run()
IJ.run("Collect Garbage") # Clean memory

print "\nProcess has finished"

filename_csv_all = "tracks_all.csv"
path_individual_movies_tracks = os.path.join(output_path, "individual_movies_tracks")
combine_csv_tracks(path_individual = path_individual_movies_tracks, path_all = output_path, filename_csv_all = filename_csv_all)

###################################################



