"""
Script:    Colocalization_GUI
Version:   3.0.0
Author:    Eric Kramer i Rosado

Description
-----------
    From a list of bona fide C1 tracks, check the colocalization with C2.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Standard libraries
import os
import csv

# ImageJ
import ij
from ij import IJ, WindowManager
from ij.io import Opener

from ij.measure import ResultsTable, Calibration
from ij.plugin import ImageCalculator
from ij.gui import NonBlockingGenericDialog, Roi, OvalRoi, Overlay
from fiji.util.gui import GenericDialogPlus
from collections import defaultdict

# For TrackMate
from fiji.plugin.trackmate import TrackMate, Model, Logger, Settings, SelectionModel
from fiji.plugin.trackmate.detection import DogDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
from fiji.plugin.trackmate.features import FeatureFilter

### For GUI
from fiji.plugin.trackmate.visualization.hyperstack import HyperStackDisplayer
from fiji.plugin.trackmate.gui.wizard import TrackMateWizardSequence
from fiji.plugin.trackmate.gui import GuiUtils
from fiji.plugin.trackmate.gui.wizard.descriptors import ConfigureViewsDescriptor
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.gui.displaysettings.DisplaySettings import TrackMateObject

# Java
from java.awt               import Color, FlowLayout, Dimension, Toolkit, Window
from java.util.concurrent   import CountDownLatch
from javax.swing            import JFrame, JDialog, JPanel, JButton, JRadioButton, ButtonGroup, BoxLayout, JLabel, JTextField

#@ PrefService prefs

########################################################
#################### ALL FUNCTIONS #####################
########################################################

def crop_from_coordinates(whole_FOV, crop_size, x_center, y_center, track_start, track_stop, extra_frames):
    """
    Crop a spatial-temporal region from the full field-of-view image stack centered around given coordinates.
    
    The crop is centered on (x_center, y_center) with size `crop_size`, but will be adjusted if near edges.
    The temporal crop covers from `track_start - extra_frames` to `track_stop + extra_frames` frames, clamped within valid frame range.
    
    Parameters:
        whole_FOV (ij.ImagePlus): The full image stack to crop from.
        crop_size (int): Desired crop size in pixels (width and height).
        x_center (int): X-coordinate center for spatial crop.
        y_center (int): Y-coordinate center for spatial crop.
        track_start (int): Starting frame index of the track.
        track_stop (int): Ending frame index of the track.
        extra_frames (int): Number of extra frames to include before and after the track.
    
    Returns:
        tuple:
            crop (ij.ImagePlus): The cropped image stack.
            crop_start (int): Starting frame of the crop.
            crop_stop (int): Ending frame of the crop.
            x0 (int): X-coordinate of the crop’s top-left corner.
            y0 (int): Y-coordinate of the crop’s top-left corner.
    
    Raises:
        ValueError: If inputs are invalid or cropping results in empty image.
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
    crop_start = max(0, track_start - extra_frames)
    crop_stop = min(Nframes, track_stop + extra_frames)
    duration = crop_stop - crop_start
    if duration <= 0:
        raise ValueError("Duration of crop is non-positive. Check track frames and extra_frames.")

    # Create crop in both space and time
    stack = whole_FOV.getStack()
    crop_stack = stack.crop(x0, y0, crop_start, int(round(adjusted_crop_x)), int(round(adjusted_crop_y)), duration)
    
    if crop_stack.getSize() == 0:
        raise RuntimeError("The crop operation returned an empty stack. Likely due to out-of-bounds coordinates. "
                        "crop_start: %d, duration: %d, stack size: %d" % (crop_start, duration, stack.getSize()))

    # Give a title to the crop, based on its TrackMate parameters
    crop_title = "crop_X_%d_Y_%d_from_%d_to_%d.tif" % (x_center, y_center, track_start, track_stop)
    # Transform the stack object into an ImagePlus object
    crop = ij.ImagePlus(crop_title, crop_stack)
    
    return crop, crop_start, crop_stop, x0, y0

def combine_C1_C2_gaussians(rawC1, rawC2, track_C1_start, track_C1_stop, crop_C1_start, zoom = 1500, move_roiC2_x = 0, move_roiC2_y = 0, ROI_diameter = 10):
    """
    Apply a 3D Gaussian filter to two raw video stacks (C1 and C2), combine them side by side,
    mark regions of interest (ROIs) on each, and display the combined image with zoom.

    Parameters:
        rawC1 (ij.ImagePlus): Raw image stack for channel 1.
        rawC2 (ij.ImagePlus): Raw image stack for channel 2.
        track_C1_start (int): Start frame index of the C1 track (absolute frame number).
        track_C1_stop (int): Stop frame index of the C1 track (absolute frame number).
        crop_C1_start (int): Starting frame of the cropped C1 stack (to align frame indices).
        zoom (int, optional): Zoom factor for display. Defaults to 1500.
        move_roiC2_x (int, optional): Horizontal shift for C2 ROI. Defaults to 0.
        move_roiC2_y (int, optional): Vertical shift for C2 ROI. Defaults to 0.
        ROI_diameter (int, optional): Diameter of the oval ROI in pixels. Defaults to 10.
    Returns:
        ij.ImagePlus: The combined and processed image stack with overlays.
    """
    
    # Duplicate and apply 3D Gaussian blur to C1 stack
    rawC1.show()
    gaussianC1 = rawC1.duplicate()
    IJ.run(gaussianC1, "Gaussian Blur 3D...", "x=0.5 y=0.5 z=1.5")                        
    gaussianC1.setTitle("Gaussian_C1") 
    gaussianC1.show()
       
    # Duplicate and apply 3D Gaussian blur to C2 stack
    rawC2.show()
    gaussianC2 = rawC2.duplicate()
    IJ.run(gaussianC2, "Gaussian Blur 3D...", "x=0.5 y=0.5 z=1.5")
    gaussianC2.setTitle("Gaussian_C2")
    gaussianC2.show()

    # Close the raw stacks to free memory
    rawC1.close()
    rawC2.hide()

    # Combine C1 and C2 stacks side by side
    IJ.run("Combine...", "stack1=" + gaussianC1.getTitle() + " stack2=" + gaussianC2.getTitle())

    # Retrieve the ID list of open images and get the combined image
    imageIDs = WindowManager.getIDList()
    combined = WindowManager.getImage(imageIDs[-1]) # Get the most recent image (the combined one)
    
    # Rename the combined image with its dimensions
    combined.setTitle("C1 vs. C2 (gaussian)")
    
    # Set the desired zoom level and enhance the contrast
    IJ.run(combined, "Set... ", "zoom=%d x=5 y=5"%(zoom))
    IJ.run("Enhance Contrast", "saturated=0.35")
    
    combined_width, combined_height = combined.getWidth(), combined.getHeight()
    # Calculate frame range relative to crop start
    start = track_C1_start - crop_C1_start
    end   = track_C1_stop - crop_C1_start

    # Calculate centers of C1 and C2 in the combined image (assuming side-by-side)
    x_center_C1 = combined_width/4
    y_center_C1 = combined_height/2
    x_center_C2 = combined_width *3/4
    y_center_C2 = combined_height/2
    
    # Calculate ROI top-left corners with optional offset for C2
    roi_C1_x = x_center_C1 - ROI_diameter//2
    roi_C1_y = y_center_C1 - ROI_diameter//2
    roi_C2_x = x_center_C2 - ROI_diameter//2 + move_roiC2_x
    roi_C2_y = y_center_C2 - ROI_diameter//2 + move_roiC2_y

    overlay = Overlay()
    
    # Add green ROI on C1 and yellow ROI on C2 for each frame in track range
    for frame in range(start, end + 1):
        roiC1 = OvalRoi(roi_C1_x, roi_C1_y, ROI_diameter, ROI_diameter)
        roiC1.setPosition(frame)
        roiC1.setStrokeColor(Color.GREEN)
        overlay.add(roiC1)
        
        roiC2 = OvalRoi(roi_C2_x, roi_C2_y, ROI_diameter, ROI_diameter)
        roiC2.setPosition(frame)
        roiC2.setStrokeColor(Color.YELLOW)
        overlay.add(roiC2)
    
    combined.setOverlay(overlay)
    IJ.run("Tile") # Tile all open images for better visibility
    
    return combined

###################################################################################
###################################################################################
###################################################################################

class NonBlockingCurationDialog(JDialog):
    def __init__(self, latch, location = (100,100), title="Colocalization check"):
        # Create a non-modal dialog so that other windows remain accessible.
        JDialog.__init__(self)
        self.setTitle(title)
        self.setModal(False)
        self.latch = latch
        
        # Use a vertical BoxLayout for stacking components
        self.getContentPane().setLayout(BoxLayout(self.getContentPane(), BoxLayout.Y_AXIS))
        
        message_label = JLabel("Is there colocalization in C2?")
        message_panel = JPanel(FlowLayout(FlowLayout.LEFT))
        message_panel.add(message_label)
        self.getContentPane().add(message_panel)
        
        # Buttons
        self.radio1 = JRadioButton("Yes, perform tracking")
        self.radio2 = JRadioButton("Yes, but it's a bad track")
        self.radio3 = JRadioButton("No colocalization")
        self.radio4 = JRadioButton("Discard (bad track in C1)")

        self.group = ButtonGroup()
        for rb in [self.radio1, self.radio2, self.radio3, self.radio4]:
            self.group.add(rb)
            panel = JPanel(FlowLayout(FlowLayout.LEFT))
            panel.add(rb)
            self.getContentPane().add(panel)
        
        # Notes field
        notesPanel = JPanel(FlowLayout(FlowLayout.LEFT))
        notesPanel.add(JLabel("Notes:"))
        self.notesField = JTextField(15)
        notesPanel.add(self.notesField)
        self.getContentPane().add(notesPanel)
        
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
        self.setPreferredSize(Dimension(280, 350))
        self.pack()
        self.setLocation(location[0], location[1])  # Position the dialog at a specific location
        
        # Store (label, stop_flag, notes)
        self.selectionResult = None
           
    def onOk(self):
        
        if self.radio1.isSelected():
            label = "colocalize"
        elif self.radio2.isSelected():
            label = "discarded_in_C2"
        elif self.radio3.isSelected():
            label = "no_colocalization"
        elif self.radio4.isSelected():
            label = "discarded_in_C1"
        else:
            label = None
        
        notes = self.notesField.getText()
        self.selectionResult = (label, False, notes)
        self.setVisible(False)
        self.latch.countDown()
        
    def onStop(self):
        self.selectionResult = (None, True, None)
        self.setVisible(False)
        self.latch.countDown()

    def getSelection(self):
        return self.selectionResult


def show_gui_colocalization(location_gui=(100, 100)):
    """
    Show a GUI dialog to annotate colocalization manually between C1 and C2.
    
    Returns:
        tuple(str label, str notes):
            label: One of ["colocalize", "discarded_in_C2", "no_colocalization", "discarded_in_C1"]
            notes: User-entered notes, or "-" if empty.
    """

    latch = CountDownLatch(1)
    dlg = NonBlockingCurationDialog(latch, location = location_gui, title="Colocalization check")
    dlg.setVisible(True)  # This call blocks until the dialog is hidden
    latch.await()         # Block here until OK or Stop is clicked
    
    # Retrieve values from the dialog
    selection = dlg.getSelection()

    if selection is None:
        print("Colocalization Dialog was closed unexpectedly.")
        IJ.run("Close All")
        exit()
    
    label, stop_bool, initial_notes = selection
    
    if stop_bool:
        print("Annotation stopped by user.")
        IJ.run("Close All")
        exit()
    
    if initial_notes is None or len(initial_notes.strip()) == 0:
        initial_notes = "-"
        
    # Log user selection
    if label == "colocalize":
        print("Colocalization visually detected.")
    elif label == "discarded_in_C2":
        print("Colocalization detected, but discarded in C2.")
    elif label == "no_colocalization":
        print("No colocalization observed.")
        IJ.run("Close All")
    elif label == "discarded_in_C1":
        print("Track discarded due to C1 issues.")
        IJ.run("Close All")
    else:
        print("No valid selection made.")
        IJ.run("Close All")
        exit()
        
    return label, initial_notes
    

def preprocess(imp):
    """
    Preprocess the input image:
      - Set calibration (pixel dimensions, frame interval).
      - Verify image dimensions and calibration.
      - Apply a 3D Gaussian blur.
      - Generate a Difference-of-Gaussian (DoG) image.
      
    Returns a tuple: (processed image, DoG image)
    """
    
    print("Preprocessing C2...")

    # Set calibration parameters
    TimeUnit = "unit"
    newCal = Calibration()
    newCal.pixelWidth = 1
    newCal.pixelHeight = 1
    newCal.frameInterval = 1
    newCal.setXUnit("pixel")
    newCal.setYUnit("pixel")
    newCal.setTimeUnit(TimeUnit)

    imp.setCalibration(newCal)
    cal = imp.getCalibration()
    
    # Verify image dimensions and calibration
    width = imp.getWidth()
    height = imp.getHeight()
    num_frames = imp.getNFrames()
    print("\tImage dimensions: width=%d, height=%d, frames=%d" % (width, height, num_frames))
    
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid image dimensions: width or height is zero or negative!")
    if cal.pixelWidth <= 0 or cal.pixelHeight <= 0:
        raise RuntimeError("Invalid calibration: pixel dimensions must be positive!")
    if num_frames < 1:
        raise RuntimeError("Invalid image: no frames found!")
        
    # Apply a 3D Gaussian blur to reduce noise
    IJ.run(imp, "Gaussian Blur 3D...", "x=0.5 y=0.5 z=2")                        
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
    
    return (imp_processed, imp_DoG)
    

def run_TM_C2(raw_C2, crop_C2_start, crop_C2_stop, track_C1_start, track_C1_stop, 
              zoom = 1500, move_roiC2_x = 0, move_roiC2_y = 0, trackmate_ROI_size = 20, 
              location_gui = (100,100)):
    """
    Run TrackMate on a region of interest within channel 2 (C2) around a given location.
    
    This function duplicates and preprocesses the raw C2 image stack, sets up TrackMate
    with appropriate detectors and trackers, applies spatial spot filters to focus on
    the ROI, and launches TrackMate's GUI wizard for user interaction. After tracking,
    it prompts the user to select a track index or indicate that no track was detected.

    Parameters:
    -----------
    raw_C2 : ImagePlus
        The raw image stack of channel 2 to analyze.
    crop_C2_start : int
        Frame index marking the start of the cropped C2 stack.
    crop_C2_stop : int
        Frame index marking the end of the cropped C2 stack.
    track_C1_start : int
        Start frame of the C1 track to compare with.
    track_C1_stop : int
        End frame of the C1 track.
    zoom : int, optional
        Zoom level for display (default: 1500).
    move_roiC2_x : int, optional
        Horizontal pixel offset to adjust ROI center in C2 (default: 0).
    move_roiC2_y : int, optional
        Vertical pixel offset to adjust ROI center in C2 (default: 0).
    trackmate_ROI_size : int, optional
        Size in pixels of the square ROI for TrackMate analysis (default: 20).
    location_gui : tuple(int, int), optional
        Screen location (x,y) for the TrackMate selection dialog (default: (100, 100)).

    Returns:
    --------
    tuple:
        - FeatureModel object from TrackMate (fm)
        - Selected track index (int or None)
        - Boolean indicating if "No track detected" was checked
        - TrackMate Model object
        - User notes string
        - Dict of tracking metadata (tracker settings, detector settings, track filters)
    """
    
    # Duplicate and preprocess the C2 stack
    gaussian_C2 = raw_C2.duplicate()
    gaussian_C2.show()
    gaussian_C2.setDimensions(1, 1, gaussian_C2.getStackSize())  # channels, slices, frames
    gaussian_C2, DoG_C2 = preprocess(gaussian_C2)
    gaussian_C2.setTitle("Crop C2 Gaussian for TrackMate")

    IJ.run(gaussian_C2, "Set... ", "zoom=%d"%(zoom))
    IJ.run(gaussian_C2, "Enhance Contrast", "saturated=0.35")
    
    # Prepare and run TrackMate analysis on the selected ROI
    print("Running TrackMate on C2")
    FOV_to_track = gaussian_C2
    cal = FOV_to_track.getCalibration()
    dims = FOV_to_track.getDimensions()

    # Start the tracking
    model = Model()
    #Read the image calibration
    model.setPhysicalUnits(cal.getUnit(), cal.getTimeUnit())
    # Send all messages to ImageJ log window
    model.setLogger(Logger.IJ_LOGGER)
    
    # Configure detector settings
    settings = Settings(FOV_to_track)
    settings.detectorFactory = DogDetectorFactory()
    settings.detectorSettings = {
        'RADIUS': 1.5, 
        'TARGET_CHANNEL': 1, 
        'THRESHOLD': float(INIT_Q), 
        'DO_SUBPIXEL_LOCALIZATION': True, 
        'DO_MEDIAN_FILTERING': False,
    }
    
    width, height = gaussian_C2.getWidth(), gaussian_C2.getHeight()
    x_center_C2, y_center_C2 = width/2, height/2

    # Displacement applied by the user
    x_roi_center = x_center_C2 + move_roiC2_x
    y_roi_center = y_center_C2 + move_roiC2_y
    
    # Compute top-left of bounding box of the circle
    roi_xmin = int(x_roi_center - trackmate_ROI_size/2)
    roi_ymin = int(y_roi_center - trackmate_ROI_size/2)
    roi_xmax = roi_xmin + trackmate_ROI_size
    roi_ymax = roi_ymin + trackmate_ROI_size
    
    startC1 = track_C1_start - crop_C2_start
    endC1   = track_C1_stop - crop_C2_start
    
    # Draw the circular ROI on the image
    WindowManager.setCurrentWindow(gaussian_C2.getWindow())

    ############################################
    ############################################
    ############################################
    
    # Apply spot detection filters to narrow down the ROI for TrackMate
    settings.addSpotFilter(FeatureFilter('POSITION_X', roi_xmin, True))
    settings.addSpotFilter(FeatureFilter('POSITION_X', roi_xmax, False))
    settings.addSpotFilter(FeatureFilter('POSITION_Y', roi_ymin, True))
    settings.addSpotFilter(FeatureFilter('POSITION_Y', roi_ymax, False))

    # Configure the tracker settings
    settings.trackerFactory = SparseLAPTrackerFactory()
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
    
    settings.trackerSettings['LINKING_MAX_DISTANCE'] = LINKING_MAX_DISTANCE
    settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = ALLOW_TRACK_SPLITTING
    settings.trackerSettings['SPLITTING_MAX_DISTANCE'] = SPLITTING_MAX_DISTANCE
    settings.trackerSettings['ALLOW_TRACK_MERGING'] = ALLOW_TRACK_MERGING
    settings.trackerSettings['MERGING_MAX_DISTANCE'] = MERGING_MAX_DISTANCE
    settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = GAP_CLOSING_MAX_DISTANCE
    settings.trackerSettings['MAX_FRAME_GAP'] = MAX_FRAME_GAP
    
    # Configure all analyzers known to TrackMate
    settings.addAllAnalyzers()
    
    # Add filters for track selection
    settings.addTrackFilter(FeatureFilter('TRACK_DURATION', 1/time_interval, True))
    settings.addTrackFilter(FeatureFilter('TRACK_MEAN_QUALITY', float(MEAN_Q), True))
    
    # Instantiate plugin
    trackmate = TrackMate(model, settings)
    if not trackmate.checkInput():
        raise RuntimeError("TrackMate input error: " + str(trackmate.getErrorMessage()))
    if not trackmate.process():
        error_msg = trackmate.getErrorMessage()
    
        # Handle the common case: no spots detected
        if "spot collection is empty" in error_msg.lower():
    
            msg = (
                "TrackMate could not detect any spots in C2.\n\n"
                "This usually means the signal is too dim for the current thresholds.\n\n"
                "Current parameters:\n"
                "  INIT_Q = %.2f\n"
                "  MEAN_Q = %.2f\n\n"
                "Suggestion:\n"
                "Lower INIT_Q and/or MEAN_Q in the GUI and try again.\n"
                "Typical values for dim signals:\n"
                "  INIT_Q: 1.5-2.5\n"
                "  MEAN_Q: 2.0-3.0"
            ) % (INIT_Q, MEAN_Q)
    
            IJ.showMessage("TrackMate: No spots detected", msg)
    
            return None  # Skip this track
    
        # Any other TrackMate error
        raise RuntimeError("TrackMate processing error: " + str(error_msg))
    
    # Display the tracking results
    FOV_to_track.show()
    model.getLogger().log("Found %d tracks"%(model.getTrackModel().nTracks(True)))
    selectionModel = SelectionModel(model)
    ds = DisplaySettingsIO.readUserDefault()
    ds.setTrackColorBy(TrackMateObject.TRACKS, "TRACK_INDEX")
    ds.setSpotColorBy(TrackMateObject.TRACKS, "TRACK_INDEX")
    displayer =  HyperStackDisplayer(model, selectionModel, FOV_to_track, ds)
    displayer.render()
    displayer.refresh()
    
    IJ.run("Tile")
    
    overlay = FOV_to_track.getOverlay()
    if overlay is None:
        overlay = Overlay()

    for frame in range(1, FOV_to_track.getNFrames() + 1):
        roi = Roi(roi_xmin, roi_ymin, trackmate_ROI_size, trackmate_ROI_size)
        roi.setPosition(frame)
        if frame in range(startC1, endC1 + 1):
            roi.setStrokeColor(Color.GREEN)
        else:
            roi.setStrokeColor(Color.YELLOW)
        overlay.add(roi)
    FOV_to_track.setOverlay(overlay)

    # Launch the TrackMate Wizard for user interaction
    sequence = TrackMateWizardSequence(trackmate, selectionModel, ds)
    guiState = ConfigureViewsDescriptor.KEY
    sequence.setCurrent(guiState)
    frame = sequence.run("TrackMate importing CSV")
    GuiUtils.positionWindow(frame, FOV_to_track.getWindow())
    frame.setVisible(True)
    
    IJ.run("Tile")
    
    # Ask the user for the track index that colocalizes with C1
    gui_index = NonBlockingGenericDialog("Track Index")
    gui_index.addNumericField("Index of selected track", 0, 0)
    gui_index.addCheckbox("No track detected", False)
    gui_index.addStringField("Notes:", "")
    gui_index.setLocation(*location_gui)
    gui_index.showDialog()
    
    if gui_index.wasOKed():
        try:
            selected_id = int(gui_index.getNextNumber())
        except:
            selected_id = None
        no_track_detected = gui_index.getNextBoolean()
        final_notes = gui_index.getNextString()
        
        IJ.run("Close All")
        frame.setVisible(False)

        # Return the relevant track data
        fm = model.getFeatureModel()
        
        final_settings = trackmate.getSettings()
        final_tracker_settings = final_settings.trackerSettings.copy()
        final_detector_settings = final_settings.detectorSettings.copy()

        final_track_filters = {
            f.feature: f.value
            for f in final_settings.getTrackFilters()
        }

        tracking_metadata = {
            "tracker_settings": final_tracker_settings,
            "detector_settings": final_detector_settings,
            "track_filters": final_track_filters
        }
       
        return (fm, selected_id, no_track_detected, model, final_notes, tracking_metadata)

#################################################################################
#################################################################################
#################################################################################

def write_spots(fm, model, selected_id, path_file_spots_modified, crop_C2_start, crop_C2_xcorner, crop_C2_ycorner):
    """
    This function writes the details of the spots from the selected C2 track to a CSV file.

    Parameters:
    -----------
    fm : FeatureModel
        The TrackMate feature model providing access to spot and track features.
    model : Model
        The TrackMate model object containing tracks and spots.
    selected_id : int
        The ID of the selected track whose spots will be written.
    path_file_spots_modified : str
        Path to the CSV file where spot data will be saved.
    crop_C2_start : int
        Frame offset to correct spot frame numbers relative to the full video.
    crop_C2_xcorner : float
        X-coordinate offset to correct spot X positions relative to the full video.
    crop_C2_ycorner : float
        Y-coordinate offset to correct spot Y positions relative to the full video.
    """
    
    try:
        with open(path_file_spots_modified, "w") as spotfile:
    
            # Write column names as header in the CSV file
            column_names = "ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,FRAME,MEAN_INTENSITY_CH2,MEDIAN_INTENSITY_CH2,MIN_INTENSITY_CH2,MAX_INTENSITY_CH2,STD_INTENSITY_CH2,CONTRAST_CH2,SNR_CH2\n"
            spotfile.write(column_names)
            
            # Get the track spots for the specified track ID
            track_spots = model.getTrackModel().trackSpots(selected_id)
            
            for spot in track_spots:
                data = [
                    float(spot.ID()),
                    float(fm.getTrackFeature(selected_id, "TRACK_ID")),
                    spot.getFeature("QUALITY"), 
                    spot.getFeature("POSITION_X") + crop_C2_xcorner, 
                    spot.getFeature("POSITION_Y") + crop_C2_ycorner,
                    spot.getFeature("FRAME") + crop_C2_start, 
                    spot.getFeature("MEAN_INTENSITY_CH1"), 
                    spot.getFeature("MEDIAN_INTENSITY_CH1"), 
                    spot.getFeature("MIN_INTENSITY_CH1"),
                    spot.getFeature("MAX_INTENSITY_CH1"), 
                    spot.getFeature("STD_INTENSITY_CH1"), 
                    spot.getFeature("CONTRAST_CH1"), 
                    spot.getFeature("SNR_CH1")
                ]
    
                data = ["{:.6f}".format(x) if isinstance(x, float) else str(x) for x in data]  # Ensure 6 decimal places for float elements
                # Write the data as a line in the CSV file
                data_to_write = ",".join(data) + "\n"    
                spotfile.write(data_to_write)                                           
    
    except IOError as e:
        print("Error writing to file {}: {}".format(path_file_spots_modified, e))


def write_tracks(fm, tracking_metadata, selected_id, experiment, file_id, writer, colocalize_id, 
                 notes, crop_C2_start, crop_C2_stop, crop_C2_xcorner, crop_C2_ycorner, time_interval):

    """
    Writes C2 track information to a CSV file.

    Parameters:
    -----------
    fm : FeatureModel
        TrackMate feature model with track features.
    tracking_metadata : dict
        Dictionary containing detector, tracker settings and track filters.
    selected_id : int
        The track ID to write.
    experiment : str
        Experiment identifier or name.
    file_id : str or int
        Identifier for the file or field of view.
    writer : csv.DictWriter
        CSV writer object with predefined fieldnames.
    colocalize_id : int or str
        Identifier linking C1 and C2 tracks for colocalization.
    notes : str
        Notes or comments for the track.
    crop_C2_start : int
        Frame offset due to cropping in C2.
    crop_C2_stop : int
        Frame index marking end of crop in C2 (not currently used here but kept for completeness).
    crop_C2_xcorner : float
        X offset due to cropping in C2.
    crop_C2_ycorner : float
        Y offset due to cropping in C2.
    time_interval : float
        Time between frames (in seconds or relevant units).
    """

    row = {col: None for col in writer.fieldnames}  # Initialize all columns with None

    row.update({
        "EXPERIMENT": experiment,
        "COLOCALIZE_ID": colocalize_id,
        "CHANNEL": 2,
        "FILE_ID": file_id,
        "NOTES": notes,
        "TRACK_ID": fm.getTrackFeature(selected_id, "TRACK_ID"),
        "TRACK_DURATION": fm.getTrackFeature(selected_id, "TRACK_DURATION"),
        "TRACK_START": fm.getTrackFeature(selected_id, "TRACK_START") + crop_C2_start,
        "TRACK_STOP": fm.getTrackFeature(selected_id, "TRACK_STOP") + crop_C2_start,
        "TRACK_DISPLACEMENT": fm.getTrackFeature(selected_id, "TRACK_DISPLACEMENT"),
        "TRACK_X_LOCATION": fm.getTrackFeature(selected_id, "TRACK_X_LOCATION") + crop_C2_xcorner,
        "TRACK_Y_LOCATION": fm.getTrackFeature(selected_id, "TRACK_Y_LOCATION") + crop_C2_ycorner,
        "TRACK_MEAN_SPEED": fm.getTrackFeature(selected_id, "TRACK_MEAN_SPEED"),
        "TRACK_MAX_SPEED": fm.getTrackFeature(selected_id, "TRACK_MAX_SPEED"),
        "TRACK_MIN_SPEED": fm.getTrackFeature(selected_id, "TRACK_MIN_SPEED"),
        "TRACK_MEDIAN_SPEED": fm.getTrackFeature(selected_id, "TRACK_MEDIAN_SPEED"),
        "TRACK_STD_SPEED": fm.getTrackFeature(selected_id, "TRACK_STD_SPEED"),
        "TRACK_MEAN_QUALITY": fm.getTrackFeature(selected_id, "TRACK_MEAN_QUALITY"),
        "MAX_DISTANCE_TRAVELED": fm.getTrackFeature(selected_id, "MAX_DISTANCE_TRAVELED"),
        "TRACK_MEAN_Q_ENV": fm.getTrackFeature(selected_id, "TRACK_MEAN_Q_ENV"),
        "TRACK_MEAN_Q_IN": fm.getTrackFeature(selected_id, "TRACK_MEAN_Q_IN"),
        "TRACK_MEAN_Q_IN_BEFORE": fm.getTrackFeature(selected_id, "TRACK_MEAN_Q_IN_BEFORE"),
        "TRACK_MEAN_Q_IN_AFTER": fm.getTrackFeature(selected_id, "TRACK_MEAN_Q_IN_AFTER"),
        "CORRELATION_FACTOR": fm.getTrackFeature(selected_id, "CORRELATION_FACTOR"),
        "CONFINEMENT_RATIO": fm.getTrackFeature(selected_id, "CONFINEMENT_RATIO"),

        # Tracking parameters
        "INIT_Q": tracking_metadata["detector_settings"]["THRESHOLD"],
        "MEAN_Q": tracking_metadata["track_filters"]["TRACK_MEAN_QUALITY"],
        "TIME_INTERVAL": time_interval,

        "ALLOW_TRACK_SPLITTING": tracking_metadata["tracker_settings"]["ALLOW_TRACK_SPLITTING"],
        "SPLITTING_MAX_DISTANCE": tracking_metadata["tracker_settings"]["SPLITTING_MAX_DISTANCE"],
        "ALLOW_TRACK_MERGING": tracking_metadata["tracker_settings"]["ALLOW_TRACK_MERGING"],
        "MERGING_MAX_DISTANCE": tracking_metadata["tracker_settings"]["MERGING_MAX_DISTANCE"],
        
        "LINKING_MAX_DISTANCE": tracking_metadata["tracker_settings"]["LINKING_MAX_DISTANCE"],
        "GAP_CLOSING_MAX_DISTANCE": tracking_metadata["tracker_settings"]["GAP_CLOSING_MAX_DISTANCE"],
        "MAX_FRAME_GAP": tracking_metadata["tracker_settings"]["MAX_FRAME_GAP"],
    })

    writer.writerow(row)
    
###########################################################################

def open_image(path):
    imp = None
    try:
        imp = Opener.openUsingBioFormats(path)
        if imp is None:
            raise RuntimeError("BioFormats returned None")
    except Exception as e:
        print("BioFormats failed for %s: %s" % (path, e))
        print("Trying IJ.openImage()...")
        imp = IJ.openImage(path)

    if imp is None:
        raise RuntimeError("Could not open image: %s" % path)

    return imp

###########################################################################

def load_csv(path):
    """
    Load a CSV file into ImageJ ResultsTable.
    Args:
        path_csv_full (str): Full path to the CSV file
    Returns:
        TM_table: the ResultsTable object
    """
    if not os.path.exists(path):
        raise IOError("CSV file not found: %s" % path)
    
    # Open CSV in ImageJ
    IJ.open(path)
    TM_table = ResultsTable.getResultsTable(os.path.basename(path))
    
    if TM_table is None or TM_table.size() == 0:
        raise ValueError("No data found in CSV file: " + os.path.basename(path))
        
    return TM_table
    
###########################################################################

def load_already_curated(path, channel = "1"):
    """
    Load previously curated tracks and return a set of (FILE_ID, TRACK_ID) tuples.

    Args:
        path (str): full path to the curated CSV
        channel (str): channel to filter for (default "1")

    Returns:
        already_curated (set of tuples): {(FILE_ID, TRACK_ID), ...}
        rows_in_csv_out (list of dicts): raw rows from CSV
    """
    
    rows_in_csv_out = []

    if os.path.exists(path) and os.stat(path).st_size > 0:
        with open(path, "r") as f:
            rows_in_csv_out = list(csv.DictReader(f))

    # Build a set of curated tracks for the specified channel
    already_curated = set()
    for r in rows_in_csv_out:
        try:
            if r.get("CHANNEL") == str(channel):
                file_id = int(r["FILE_ID"])
                track_id = int(r["TRACK_ID"])
                already_curated.add((file_id, track_id))
        except (KeyError, ValueError):
            # skip rows with missing or invalid data
            continue

    return already_curated, rows_in_csv_out

###########################################################################

def group_uncurated_tracks(TM_table, already_curated):
    
    tracks_by_file = defaultdict(list)
    
    for i in range(TM_table.size()):
        file_id = int(TM_table.getStringValue("FILE_ID", i))
        track_id = int(TM_table.getStringValue("TRACK_ID", i))
        
        if (file_id, track_id) not in already_curated:
            tracks_by_file[file_id].append(i)
            
    return tracks_by_file

###########################################################################   

def get_track_data(TM_table, i, column_names):
    
    track_data = {col: TM_table.getStringValue(col, i) for col in column_names}
    
    for key in ["FILE_ID", "TRACK_ID", "TRACK_START", "TRACK_STOP"]:
        track_data[key] = int(track_data[key])
    for key in ["TRACK_X_LOCATION", "TRACK_Y_LOCATION"]:
        track_data[key] = float(track_data[key])
        
    return track_data
         
###########################################################################           

def run_everything():
    """
    Process C1 tracks, ask the user to curate colocalization in C2, run TrackMate if needed,
    and write both C1- and C2-derived rows into a single CSV.
    """
    # Open the CSV file containing C1 curated tracks
    path_csv_C1 = os.path.join(path_csv, filename_csv_C1)
    TM_table = load_csv(path_csv_C1)
    
    # Get column headers and number of tracks
    column_names = list(TM_table.getHeadings())
    num_tracks = TM_table.size()
    
    # Load already-curated tracks
    path_csv_out = os.path.join(path_csv, filename_csv_colocalizing)
    already_curated, rows_in_csv_out = load_already_curated(path_csv_out)
    
    print "\n*************************************"
    print "\tNumber of tracks:", num_tracks
    print "\tAlready curated:", len(already_curated)
    print "*************************************\n"
    
    # Group uncurated tracks by FILE_ID
    tracks_by_file = group_uncurated_tracks(TM_table, already_curated)

    # Define column names
    mandatory_columns = ["EXPERIMENT", "COLOCALIZE_ID", "CHANNEL", "FILE_ID", "NOTES"]
    trackmate_features = [
        "TRACK_ID", "TRACK_DURATION", "TRACK_START", "TRACK_STOP", 
        "TRACK_X_LOCATION", "TRACK_Y_LOCATION", "TRACK_DISPLACEMENT", "TRACK_MEAN_SPEED", "TRACK_MAX_SPEED",
        "TRACK_MIN_SPEED", "TRACK_MEDIAN_SPEED", "TRACK_STD_SPEED", "TRACK_MEAN_QUALITY",
        "MAX_DISTANCE_TRAVELED", "TRACK_MEAN_Q_ENV", "TRACK_MEAN_Q_IN", 
        "TRACK_MEAN_Q_IN_BEFORE", "TRACK_MEAN_Q_IN_AFTER", "CORRELATION_FACTOR", 
        "CONFINEMENT_RATIO"
    ]
    trackmate_settings = [
        "INIT_Q", "MEAN_Q", "TIME_INTERVAL",
        "ALLOW_TRACK_SPLITTING", "SPLITTING_MAX_DISTANCE", "ALLOW_TRACK_MERGING", "MERGING_MAX_DISTANCE",
        "LINKING_MAX_DISTANCE", "GAP_CLOSING_MAX_DISTANCE", "MAX_FRAME_GAP"
    ]
        
    # any columns in C1 not in mandatory_columns or trackmate_features
    extras = [c for c in column_names if c not in mandatory_columns and c not in trackmate_features and c not in trackmate_settings]
    
    # Header: mandatory → trackmate → extra
    header = mandatory_columns + trackmate_features + trackmate_settings + extras
    
    ##############################################
    ############# Prepare output CSV #############
    ##############################################
    
    new_file = not os.path.exists(path_csv_out) or os.stat(path_csv_out).st_size == 0
    with open(path_csv_out, "ab") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=header)
        if new_file:
            writer.writeheader()
            
        # figure out next colocalize_id
        existing_ids = [int(row["COLOCALIZE_ID"]) for row in rows_in_csv_out if "COLOCALIZE_ID" in row]
        colocalize_id = max(existing_ids) + 1 if existing_ids else 0
        
        ##############################################
        ############## Open each movie ###############
        ##############################################
        
        # Process each file (FOV) that has tracks in the CSV
        for file_id, track_indices in sorted(tracks_by_file.items()):
            print("\n")
            print("Processing FILE_ID %d with %d uncurated tracks"%(file_id, len(track_indices)))
            # Generate the filenames for the whole FOVs of C1 and C2
            filename_wholeFOV_C1 = experiment + "_" + extra_name_FOV_C1 + "_" + str(file_id) + ".tif"
            filename_wholeFOV_C2 = experiment + "_" + extra_name_FOV_C2 + "_" + str(file_id) + ".tif"
            print "FOV_C1:", filename_wholeFOV_C1
            print "FOV_C2:", filename_wholeFOV_C2
            
            whole_FOV_C1 = open_image(os.path.join(path_movies, filename_wholeFOV_C1))
            whole_FOV_C2 = open_image(os.path.join(path_movies, filename_wholeFOV_C2))
            
            ##############################################
            ############ Process each track ##############
            ##############################################
            
            # Loop through all the tracks, from all movies
            for i in track_indices:
                
                track_data = get_track_data(TM_table, i, column_names)
                track_data["EXPERIMENT"] = str(experiment)
                
                print("\nC1 track info - start=%d - stop=%d - x=%.2f - y=%.2f"%(
                        track_data["TRACK_START"], track_data["TRACK_STOP"],
                        track_data["TRACK_X_LOCATION"], track_data["TRACK_Y_LOCATION"]
                ))
                    
                # Crop the ROI around the track in both channels
                crop_C1, crop_C1_start, crop_C1_stop, _, _, = crop_from_coordinates(
                    whole_FOV = whole_FOV_C1, crop_size = crop_size, 
                    x_center = track_data["TRACK_X_LOCATION"], y_center = track_data["TRACK_Y_LOCATION"],
                    track_start = track_data["TRACK_START"], track_stop = track_data["TRACK_STOP"],
                    extra_frames = extra_frames
                )
                                                                                    
                crop_C2, crop_C2_start, crop_C2_stop, crop_C2_xcorner, crop_C2_ycorner = crop_from_coordinates(
                    whole_FOV = whole_FOV_C2, crop_size = crop_size, 
                    x_center = track_data["TRACK_X_LOCATION"], y_center = track_data["TRACK_Y_LOCATION"], 
                    track_start = track_data["TRACK_START"], track_stop = track_data["TRACK_STOP"], 
                    extra_frames = extra_frames
                )
                
                # Combine C1 and C2 cropped images, apply Gaussian blur, and mark the ROI
                combined_C1_C2 = combine_C1_C2_gaussians(
                    rawC1 = crop_C1, rawC2 = crop_C2, 
                    track_C1_start = track_data["TRACK_START"], track_C1_stop = track_data["TRACK_STOP"], 
                    crop_C1_start = crop_C1_start, zoom = zoom, move_roiC2_x = move_roiC2_x, move_roiC2_y = move_roiC2_y
                )
                
                # Display the combined image and prompt the user to evaluate colocalization
                colocalize_label, initial_notes = show_gui_colocalization(location_gui = (location_gui_x, location_gui_y))

                # Write C1 track
                track_data_C1 = dict(track_data)
                track_data_C1.update({"COLOCALIZE_ID": colocalize_id, "CHANNEL": 1, "NOTES": "-"})
                writer.writerow(track_data_C1)
                
                # Scenario 1: discard C1 track
                if colocalize_label == "discarded_in_C1":
                    writer.writerow({
                        "EXPERIMENT": experiment,
                        "COLOCALIZE_ID": colocalize_id,
                        "CHANNEL": 2,
                        "FILE_ID": file_id,
                        "NOTES": "discarded_C1 | " + str(initial_notes)
                    })
                    IJ.run("Close All")
                    colocalize_id += 1
                    continue
                    
                # Scenario 2: confirmed colocalization, run TrackMate on C2
                elif colocalize_label == "colocalize":
                    
                    fm, selected_id, no_track_detected, model, final_notes, tracking_metadata = run_TM_C2(
                        raw_C2 = crop_C2, crop_C2_start = crop_C2_start, crop_C2_stop = crop_C2_stop,
                        track_C1_start = track_data["TRACK_START"], track_C1_stop = track_data["TRACK_STOP"], 
                        zoom = zoom, move_roiC2_x = move_roiC2_x, move_roiC2_y = move_roiC2_y, 
                        trackmate_ROI_size = trackmate_ROI_size, location_gui = (location_gui_x, location_gui_y)
                        )

                    notes = " | ".join([n for n in [initial_notes, final_notes] if n]) or "-"

                    # Scenario 2a: no good track was detected in C2
                    if no_track_detected:
                        writer.writerow({
                            "EXPERIMENT": experiment,
                            "COLOCALIZE_ID": colocalize_id,
                            "CHANNEL": 2,
                            "FILE_ID": file_id,
                            "NOTES": "no_track_detected | " + notes
                        })
                        IJ.run("Close All")
                        colocalize_id += 1
                        
                        # Get all currently open windows
                        for win in Window.getWindows():
                            if isinstance(win, JFrame) and win.isVisible():
                                title = win.getTitle()
                                if "Track tables" in title or "TrackMate" in title:
                                    win.dispose()
                        continue
                    
                    # Scenario 2b: a good track was found in C2
                    print("Selected track: %d"%(selected_id))
                    
                    # Save the modified spots and tracks for the C2 channel
                    path_file_spots_modified = os.path.join(
                        path_spots_C2, "spots_modified_" + str(colocalize_id) + ".csv"
                    )
                    
                    write_spots(
                        fm = fm, model = model, selected_id = selected_id, path_file_spots_modified = path_file_spots_modified, 
                        crop_C2_start = crop_C2_start, crop_C2_xcorner = crop_C2_xcorner, crop_C2_ycorner = crop_C2_ycorner
                    )
                    write_tracks(
                        fm = fm, tracking_metadata = tracking_metadata, selected_id = selected_id, experiment = experiment,
                        file_id = file_id, writer = writer, colocalize_id = colocalize_id, notes = notes, 
                        crop_C2_start = crop_C2_start, crop_C2_stop = crop_C2_stop, crop_C2_xcorner = crop_C2_xcorner, crop_C2_ycorner = crop_C2_ycorner,
                        time_interval = time_interval
                    )
                    
                    colocalize_id += 1
                
                # Scenario 3: there is colocalization but C2 is a bad track
                elif colocalize_label == "discarded_in_C2":
                    writer.writerow({
                        "EXPERIMENT": experiment,
                        "COLOCALIZE_ID": colocalize_id,
                        "CHANNEL": 2,
                        "FILE_ID": file_id,
                        "NOTES": "Colocalizes, but C2 discarded | " + str(initial_notes)
                    })
                    colocalize_id += 1
                
                # Scenario 4: no colocalization
                elif colocalize_label == "no_colocalization":
                    writer.writerow({
                        "EXPERIMENT": experiment,
                        "COLOCALIZE_ID": colocalize_id,
                        "CHANNEL": 2,
                        "FILE_ID": file_id,
                        "NOTES": "No colocalization | " + str(initial_notes)
                    })
                    colocalize_id += 1
                    
                # Clean-up
                for win in Window.getWindows():
                    if isinstance(win, JFrame) and win.isVisible():
                        title = win.getTitle()
                        if "Track tables" in title or "TrackMate" in title:
                            win.dispose()

                IJ.run("Close All")
                IJ.run("Collect Garbage")


################################################################
########################## USER INPUT ##########################
################################################################

def show_gui_initial():
    """
    Displays a GUI for initial user input with default values that persist between sessions.
    The dialog includes fields for:
      - Experiment name
      - Root folder (selected interactively)
      - CSV file (non-curated)
      - Display Parameters: Crop, Zoom, Extra frames (for display)
      - Naming Parameters: Extra name FOV and Extra name curated CSV.
    
    Returns a tuple:
      (experiment, root_folder, path_movies, path_csv, filename_csv_C1, filename_csv_curated, 
      crop_size, zoom, extra_frames, extra_name_FOV, extra_name_colocalizing)

    """
    # Default values
    DEFAULTS = {
        "experiment": "",
        "extra_name_FOV_C1": "prepro_C1",
        "extra_name_FOV_C2": "prepro_C2",
        "extra_name_colocalizing": "_withC2",
        "crop_size": 60,
        "zoom": 800,
        "time_interval": 0.120,
        "INIT_Q": 3.0,
        "MEAN_Q": 3.5,
        "extra_frames": 50,
        "location_gui_x": 100,
        "location_gui_y": 100,
        "move_roiC2_x": 0,
        "move_roiC2_y": 0,
        "trackmate_ROI_size": 20,
    }

    
    
    default_experiment = prefs.get(None, "experiment", DEFAULTS["experiment"])
    default_extra_name_FOV_C1 = prefs.get(None, "extra_name_FOV_C1", DEFAULTS["extra_name_FOV_C1"])
    default_extra_name_FOV_C2 = prefs.get(None, "extra_name_FOV_C2", DEFAULTS["extra_name_FOV_C2"])
    default_extra_name_colocalizing  = prefs.get(None, "extra_name_colocalizing", DEFAULTS["extra_name_colocalizing"])
    
    default_root                = prefs.get(None, "root_folder", IJ.getDirectory("home"))
    default_csv                 = prefs.get(None, "csv_file", "")
    
    default_crop_size           = prefs.getFloat(None, "crop_size", DEFAULTS["crop_size"])
    default_zoom                = prefs.getFloat(None, "zoom", DEFAULTS["zoom"])
    default_extra_frames        = prefs.getInt(None, "extra_frames", DEFAULTS["extra_frames"])

    default_location_gui_x      = prefs.getInt(None, "location_gui_x", DEFAULTS["location_gui_x"])
    default_location_gui_y      = prefs.getInt(None, "location_gui_y", DEFAULTS["location_gui_y"])
    
    default_move_roiC2_x        = prefs.getInt(None, "move_roiC2_x", DEFAULTS["move_roiC2_x"])
    default_move_roiC2_y        = prefs.getInt(None, "move_roiC2_y", DEFAULTS["move_roiC2_y"])
    default_trackmate_ROI_size  = prefs.getInt(None, "trackmate_ROI_size", DEFAULTS["trackmate_ROI_size"])
    
    default_time_interval       = prefs.getFloat(None, "time_interval", DEFAULTS["time_interval"])
    default_init_q              = prefs.getFloat(None, "INIT_Q", DEFAULTS["INIT_Q"])
    default_mean_q              = prefs.getFloat(None, "MEAN_Q", DEFAULTS["MEAN_Q"])
    
    # Create the dialog
    gui_initial = GenericDialogPlus("Input from user")
    
    # Basic information fields
    gui_initial.addStringField("Experiment name:", default_experiment)
    gui_initial.addToSameRow()
    gui_initial.addMessage("The input videos must match the following convention:\n   ExperimentName_ExtraNameFOV_FileID.tif")
    gui_initial.addDirectoryField("Root folder:", default_root, 40)
    gui_initial.addFileField("CSV file (C1 curated):", default_csv, 40)

    # Display parameters with descriptions
    gui_initial.addMessage("Display Parameters:")
    
    gui_initial.addNumericField("Crop size (for display):", default_crop_size, 0)
    gui_initial.addToSameRow()
    gui_initial.addMessage("(Crop size around the center of the track)")
    
    gui_initial.addNumericField("Zoom (for display):", default_zoom, 0)
    gui_initial.addToSameRow()
    gui_initial.addMessage("(Zoom factor for combined image display)")
    
    gui_initial.addNumericField("Extra frames (for display):", default_extra_frames, 0)
    gui_initial.addToSameRow()
    gui_initial.addMessage("(Number of frames to show before/after a track)")
    
    gui_initial.addMessage("If movies are NOT ALIGNED, move C2 ROI for tracking")
    gui_initial.addNumericField("Move X (px):", default_move_roiC2_x, 0)
    gui_initial.addNumericField("Move Y (px):", default_move_roiC2_y, 0)
    
    gui_initial.addMessage("Size ROI to perform TrackMate in C2")
    gui_initial.addNumericField("TrackMate ROI size:", default_trackmate_ROI_size, 0)
    
    screen_size = Toolkit.getDefaultToolkit().getScreenSize()
    screen_width = int(screen_size.getWidth())
    screen_height = int(screen_size.getHeight())
    gui_initial.addMessage("Screen size: {} x {}".format(screen_width, screen_height))
    gui_initial.addNumericField("Location GUI (x):", default_location_gui_x, 0)
    gui_initial.addNumericField("Location GUI (y):", default_location_gui_y, 0)
    
    # Naming parameters with descriptions
    gui_initial.addMessage("Naming Parameters:")
    gui_initial.addStringField("Extra name FOV C1", default_extra_name_FOV_C1)
    gui_initial.addToSameRow()
    gui_initial.addMessage("(Text after experiment name in input video names. Usually prepro_C1)")
    gui_initial.addStringField("Extra name FOV C2", default_extra_name_FOV_C2)
    gui_initial.addToSameRow()
    gui_initial.addMessage("(Text after experiment name in input video names. Usually prepro_C2)")
    
    gui_initial.addStringField("Extra name CSV with C2 colocalizing", default_extra_name_colocalizing)
    gui_initial.addToSameRow()
    gui_initial.addMessage("(Text to add to the csv filename with the curated tracks)")

    gui_initial.addMessage("TrackMate parameters:")
    gui_initial.addNumericField("Time interval:",  default_time_interval, 3)
    gui_initial.addNumericField("INIT_Q:", default_init_q, 2)
    gui_initial.addToSameRow()
    gui_initial.addMessage("Lower quality threshold for spot detection")
    gui_initial.addNumericField("MEAN_Q:", default_mean_q, 2)
    gui_initial.addToSameRow()
    gui_initial.addMessage("Lower mean quality threshold for track")
    gui_initial.addCheckbox("Show advanced TrackMate options", False)


    gui_initial.showDialog()

    if gui_initial.wasOKed():
        # Retrieve user inputs in order of appearance
        experiment       = gui_initial.getNextString()
        root_folder      = gui_initial.getNextString()
        csv_file         = gui_initial.getNextString()

        crop_size          = gui_initial.getNextNumber()
        zoom               = gui_initial.getNextNumber()
        extra_frames       = int(gui_initial.getNextNumber())
        
        move_roiC2_x       = int(gui_initial.getNextNumber())
        move_roiC2_y       = int(gui_initial.getNextNumber())
        trackmate_ROI_size = int(gui_initial.getNextNumber())
        
        location_gui_x     = int(gui_initial.getNextNumber())
        location_gui_y     = int(gui_initial.getNextNumber())
        
        # Check bounds
        if location_gui_x < 0 or location_gui_x > screen_width:
            print("X coordinate out of screen bounds. Resetting to default.")
            location_gui_x = default_location_gui_x
        
        if location_gui_y < 0 or location_gui_y > screen_height:
            print("Y coordinate out of screen bounds. Resetting to default.")
            location_gui_y = default_location_gui_y
            
        extra_name_FOV_C1       = gui_initial.getNextString()
        extra_name_FOV_C2       = gui_initial.getNextString()
        extra_name_colocalizing = gui_initial.getNextString()
        
        time_interval  = round(gui_initial.getNextNumber(), 3)
        INIT_Q         = round(gui_initial.getNextNumber(), 2)
        MEAN_Q         = round(gui_initial.getNextNumber(), 2)
        showAdvanced   = gui_initial.getNextBoolean()
        
        # Save the new defaults for future sessions
        prefs_dict = {
            "experiment": experiment,
            "extra_name_FOV_C1": extra_name_FOV_C1,
            "extra_name_FOV_C2": extra_name_FOV_C2,
        
            "root_folder": root_folder,
            "csv_file": csv_file,
        
            "crop_size": crop_size,
            "zoom": zoom,
            "extra_frames": extra_frames,
        
            "move_roiC2_x": move_roiC2_x,
            "move_roiC2_y": move_roiC2_y,
            "trackmate_ROI_size": trackmate_ROI_size,
        
            "extra_name_colocalizing": extra_name_colocalizing,
            "location_gui_x": location_gui_x,
            "location_gui_y": location_gui_y,
        
            "time_interval": time_interval,
            "INIT_Q": INIT_Q,
            "MEAN_Q": MEAN_Q,
        }

        for key, value in prefs_dict.items():
            prefs.put(None, key, value)

        
        # Process paths: Create sub-folders for movies and CSVs.
        root_folder = os.path.normpath(root_folder)
        path_movies = os.path.join(root_folder, "3_Preprocessed")
        path_spots_C2 = os.path.join(root_folder, "5_Analysis", "Spots_C2")
        
        # Extract CSV file name from the full path
        filename_csv_C1 = os.path.basename(csv_file)
        path_csv = os.path.dirname(csv_file)

        # Append the extra name for curated CSV; assumes original CSV extension is at least 4 characters (e.g., ".csv")
        filename_csv_colocalizing = os.path.splitext(filename_csv_C1)[0] + extra_name_colocalizing + ".csv"
        
        if not os.path.exists(root_folder):
            raise Exception("\n\n\n\nError: The root path does not exist: {0}\n\n\n\n".format(root_folder))
        if not os.path.exists(path_movies):
            raise Exception("\n\n\n\nError: The path for movies does not exist: {0}\n\n\n\n".format(path_movies))
        if not os.path.exists(path_csv):
            raise Exception("\n\n\n\nError: The path for csv does not exist: {0}\n\n\n\n".format(path_csv))
        if not os.path.exists(path_spots_C2):
            os.makedirs(path_spots_C2)
            
        params = {
            "experiment": experiment,
            "root_folder": root_folder,
            "path_movies": path_movies,
            "path_csv": path_csv,
            "path_spots_C2": path_spots_C2,
        
            "filename_csv_C1": filename_csv_C1,
            "filename_csv_colocalizing": filename_csv_colocalizing,
        
            "crop_size": crop_size,
            "zoom": zoom,
            "extra_frames": extra_frames,
        
            "move_roiC2_x": move_roiC2_x,
            "move_roiC2_y": move_roiC2_y,
            "trackmate_ROI_size": trackmate_ROI_size,
        
            "extra_name_FOV_C1": extra_name_FOV_C1,
            "extra_name_FOV_C2": extra_name_FOV_C2,
            "extra_name_colocalizing": extra_name_colocalizing,
        
            "location_gui_x": location_gui_x,
            "location_gui_y": location_gui_y,
        
            "time_interval": time_interval,
            "INIT_Q": INIT_Q,
            "MEAN_Q": MEAN_Q,
        
            "showAdvanced": showAdvanced
        
        }
        
        return params
    
    else:
        raise Exception("Initial GUI was cancelled.")


params = show_gui_initial()

experiment = params["experiment"]
root_folder = params["root_folder"]
path_movies = params["path_movies"]
path_csv = params["path_csv"]
path_spots_C2 = params["path_spots_C2"]

filename_csv_C1 = params["filename_csv_C1"]
filename_csv_colocalizing = params["filename_csv_colocalizing"]

crop_size = params["crop_size"]
zoom = params["zoom"]
extra_frames = params["extra_frames"]

move_roiC2_x = params["move_roiC2_x"]
move_roiC2_y = params["move_roiC2_y"]
trackmate_ROI_size = params["trackmate_ROI_size"]

extra_name_FOV_C1 = params["extra_name_FOV_C1"]
extra_name_FOV_C2 = params["extra_name_FOV_C2"]
extra_name_colocalizing = params["extra_name_colocalizing"]

location_gui_x = params["location_gui_x"]
location_gui_y = params["location_gui_y"]

time_interval = params["time_interval"]
INIT_Q = params["INIT_Q"]
MEAN_Q = params["MEAN_Q"]

showAdvanced = params["showAdvanced"]

################################################################
############## GUI FOR EXTRA TRACKMATE PARAMETERS ##############
################################################################
DEFAULT_ALLOW_TRACK_SPLITTING = False
DEFAULT_SPLITTING_MAX_DISTANCE = 0.0
DEFAULT_ALLOW_TRACK_MERGING = False
DEFAULT_MERGING_MAX_DISTANCE = 0.0
DEFAULT_LINKING_MAX_DISTANCE = 1.5
DEFAULT_GAP_CLOSING_MAX_DISTANCE = 2.0
DEFAULT_MAX_FRAME_GAP = 1

# Create a second dialog for advanced parameters
if showAdvanced:

    advGui = GenericDialogPlus("Advanced Tracking Parameters")

    default_allow_track_splitting = bool(prefs.getInt(None, "ALLOW_TRACK_SPLITTING", DEFAULT_ALLOW_TRACK_SPLITTING))
    default_splitting_max_distance = prefs.getFloat(None, "SPLITTING_MAX_DISTANCE", DEFAULT_SPLITTING_MAX_DISTANCE)
    default_allow_track_merging = bool(prefs.getInt(None, "ALLOW_TRACK_MERGING", DEFAULT_ALLOW_TRACK_MERGING))
    default_merging_max_distance = prefs.getFloat(None, "MERGING_MAX_DISTANCE", DEFAULT_MERGING_MAX_DISTANCE)
    default_linking_max_distance = prefs.getFloat(None, "LINKING_MAX_DISTANCE", DEFAULT_LINKING_MAX_DISTANCE)
    default_gap_closing_max_distance = prefs.getFloat(None, "GAP_CLOSING_MAX_DISTANCE", DEFAULT_GAP_CLOSING_MAX_DISTANCE)
    default_max_frame_gap = prefs.getInt(None, "MAX_FRAME_GAP", DEFAULT_MAX_FRAME_GAP)
    
    advGui.addCheckbox("ALLOW_TRACK_SPLITTING", default_allow_track_splitting)
    advGui.addNumericField("SPLITTING_MAX_DISTANCE:", default_splitting_max_distance, 2)
    advGui.addMessage(" ")
    advGui.addCheckbox("ALLOW_TRACK_MERGING", default_allow_track_merging)
    advGui.addNumericField("MERGING_MAX_DISTANCE:", default_merging_max_distance, 2)
    advGui.addMessage(" ")
    advGui.addNumericField("LINKING_MAX_DISTANCE:", default_linking_max_distance, 2)
    advGui.addNumericField("GAP_CLOSING_MAX_DISTANCE:", default_linking_max_distance, 2)
    advGui.addNumericField("MAX_FRAME_GAP:", default_max_frame_gap, 0)

    advGui.showDialog()
    
    if advGui.wasCanceled():
        print("User canceled advanced dialog")
        exit()
    else:
        ALLOW_TRACK_SPLITTING      = advGui.getNextBoolean()
        SPLITTING_MAX_DISTANCE     = round(advGui.getNextNumber(), 2)
        ALLOW_TRACK_MERGING        = advGui.getNextBoolean()
        MERGING_MAX_DISTANCE       = round(advGui.getNextNumber(), 2)
        LINKING_MAX_DISTANCE       = round(advGui.getNextNumber(), 2)
        GAP_CLOSING_MAX_DISTANCE   = round(advGui.getNextNumber(), 2)
        MAX_FRAME_GAP              = int(advGui.getNextNumber())

        # Save advanced parameters
        prefs.put(None, "ALLOW_TRACK_SPLITTING", ALLOW_TRACK_SPLITTING)
        prefs.put(None, "SPLITTING_MAX_DISTANCE", SPLITTING_MAX_DISTANCE)
        prefs.put(None, "ALLOW_TRACK_MERGING", ALLOW_TRACK_MERGING)
        prefs.put(None, "MERGING_MAX_DISTANCE", MERGING_MAX_DISTANCE)
        prefs.put(None, "LINKING_MAX_DISTANCE", LINKING_MAX_DISTANCE)
        prefs.put(None, "GAP_CLOSING_MAX_DISTANCE", GAP_CLOSING_MAX_DISTANCE)
        prefs.put(None, "MAX_FRAME_GAP", MAX_FRAME_GAP)

else:
# Set advanced parameters to defaults or previously stored values
    ALLOW_TRACK_SPLITTING       = bool(prefs.getInt(None, "ALLOW_TRACK_SPLITTING", DEFAULT_ALLOW_TRACK_SPLITTING))
    SPLITTING_MAX_DISTANCE      = round(prefs.getFloat(None, "SPLITTING_MAX_DISTANCE", DEFAULT_SPLITTING_MAX_DISTANCE), 2)
    ALLOW_TRACK_MERGING         = bool(prefs.getInt(None, "ALLOW_TRACK_MERGING", DEFAULT_ALLOW_TRACK_MERGING))
    MERGING_MAX_DISTANCE        = round(prefs.getFloat(None, "MERGING_MAX_DISTANCE", DEFAULT_MERGING_MAX_DISTANCE), 2)
    LINKING_MAX_DISTANCE        = round(prefs.getFloat(None, "LINKING_MAX_DISTANCE", DEFAULT_LINKING_MAX_DISTANCE), 2)
    GAP_CLOSING_MAX_DISTANCE    = round(prefs.getFloat(None, "GAP_CLOSING_MAX_DISTANCE", DEFAULT_GAP_CLOSING_MAX_DISTANCE), 2)
    MAX_FRAME_GAP               = prefs.getInt(None, "MAX_FRAME_GAP", DEFAULT_MAX_FRAME_GAP)

################################################################
######################## RUN EVERYTHING ########################
################################################################

IJ.run("Collect Garbage") # clean memory

run_everything()

