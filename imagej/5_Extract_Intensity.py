"""
Script:    Extract_Intensity
Version:   3.0.0
Author:    Eric Kramer i Rosado

Description
-----------
    From a list of C1-C2 bona fide events, extract the corresponding intensity profiles.
"""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import ij, os, csv
from ij.io import Opener
from ij                 import IJ
from ij.gui             import OvalRoi, Plot
from ij.measure         import ResultsTable
from fiji.util.gui      import GenericDialogPlus
from collections        import defaultdict
#@ PrefService prefs

########################################################
#################### ALL FUNCTIONS #####################
########################################################

def measure(x, y, r, t, whole_FOV, BG):
    """
    Measure the intensity of a spot in a movie at given coordinates and frame, with optional background correction
    
    Parameters:
        x, y (float): Coordinates of the spot centroid.
        r (float): Radius of the spot (in pixels).
        t (int): Time slice (frame) to measure (1-based indexing for ImageJ).
        whole_FOV (ImagePlus): Image stack (movie) in which the spot is measured.
        BG (bool): If True, perform background correction; if False, measure raw intensity.

    Returns:
    If BG=True:
        (mean_spot, mean_BG)
    If BG=False:
        (mean_x, None)
    Where:
        - mean_spot (float): Background-corrected mean intensity (BG=True).
        - mean_BG (float): Mean background intensity (BG=True only).
        - mean_x (float): Raw mean intensity (BG=False).
    """
            
    if BG:
        # First ROI: original radius
        whole_FOV.setSlice(int(t)) 
        whole_FOV.setRoi(OvalRoi(x-r+0.5, y-r+0.5, 2*r, 2*r)) 
        stats_r = whole_FOV.getStatistics()
        mean_r = stats_r.mean
        area_r = stats_r.area
        
        # Second ROI: radius + 0.5
        r_expanded = r + 0.5
        whole_FOV.setSlice(int(t)) 
        whole_FOV.setRoi(OvalRoi(x-r_expanded+0.5, y-r_expanded+0.5, 2*r_expanded, 2*r_expanded)) 
        stats_r = whole_FOV.getStatistics()
        mean_r1 = stats_r.mean
        area_r1 = stats_r.area
        
        # Third ROI: double original radius
        r_double = 2 * r                            
        whole_FOV.setSlice(int(t)) 
        whole_FOV.setRoi(OvalRoi(x-r_double+0.5, y-r_double+0.5, 2*r_double, 2*r_double)) 
        stats_r = whole_FOV.getStatistics()
        mean_2r = stats_r.mean
        area_2r = stats_r.area
        
        # Calculate background intensity
        mean_BG = (area_2r*mean_2r - area_r1*mean_r1) / (area_2r - area_r1)
        # Correct the spot intensity by subtracting the background
        mean_spot  = mean_r - mean_BG

        return mean_spot, mean_BG
        
    else:
        # If no background correction, simply measure the spot intensity directly
        whole_FOV.setSlice(int(t)) 
        whole_FOV.setRoi(OvalRoi(x-r+0.5, y-r+0.5, 2*r, 2*r)) 
        stats_r = whole_FOV.getStatistics()
        mean_x = stats_r.mean
        area_r = stats_r.area
        
        return mean_x, None
    

def filter_allspots_C1(path, C1_track_id):
    """
    From a CSV file with spots from many tracks, returns a dict mapping frame -> (x, y) for the specified TRACK_ID.
    """
    frame_to_spot = {}
    
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            
            if int(float(row["TRACK_ID"])) == C1_track_id:
                frame = int(float(row["FRAME"]))
                x = float(row["POSITION_X"])
                y = float(row["POSITION_Y"])
                frame_to_spot[frame] = (x, y)
                
        # Read the first line to get column headers
        #first_line = f.readline().strip()
        #columns = first_line.split(",")
        
        # Get indices of relevant columns
        #idx_track = columns.index("TRACK_ID")
        #idx_frame = columns.index("FRAME")
        #idx_x     = columns.index("POSITION_X")
        #idx_y     = columns.index("POSITION_Y")
        
        # Read each spot
        #for line in f:
            #elements = line.strip().split(",")
            # Extract TRACK_ID from the current line
            #track_id_from_spot = int(float(elements[idx_track]))  
            
            ## Check if this spot belongs to the specified TRACK_ID
           # if track_id_from_spot == C1_track_id:
                #frame = int(float(elements[idx_frame]))
                #x     = float(elements[idx_x])
                #y     = float(elements[idx_y])
                #frame_to_spot[frame] = (x, y)
    
    return frame_to_spot

def process_spots_C2(path):
    """
    Processes spot data from a CSV file by extracting frame, x, and y coordinates.
    Returns:
        frame_to_spot (dict[int, tuple[float, float]]): Maps frame -> (x, y)
    """
    frame_to_spot = {}
    
    with open(path, "r") as f:
        # Read the first line to get column headers
        first_line = f.readline().strip()
        columns = first_line.split(",")
        # Get indices of relevant columns
        idx_frame = columns.index("FRAME")
        idx_x     = columns.index("POSITION_X")
        idx_y     = columns.index("POSITION_Y")
        
        # Iterate through each line in the file
        for line in f:
            elements = line.strip().split(",")
            frame = int(float(elements[idx_frame]))
            x     = float(elements[idx_x])
            y     = float(elements[idx_y]) 
            frame_to_spot[frame] = (x, y)
    
    return frame_to_spot

def extract_intensity_profile(whole_FOV, frame_to_spot, num_frames, output, channel):
    """
    Extracts the intensity profile for a given track from a movie stack, optionally
    performing background correction. Missing frames are filled using the most
    recent available spot coordinates.
    
    Parameters:
        whole_FOV (ij.ImagePlus): The full field-of-view image stack from which intensities are measured.
        spots_from_track_sorted (list[list[int, float, float]]): 
            Sorted list of [frame, x, y] for the track.
        num_frames (int): Total number of frames in the movie.
        output (file object): Open file handle for writing the profile data.
        channel (int): Channel number (used for logging).
    
    Returns:
        list[float]: Intensity values at the tracked position across frames.
    """
    r = 1.5      # Radius for intensity measurement
    BG = True    # Perform background correction
    
    track_start = min(frame_to_spot.keys())
    track_stop  = max(frame_to_spot.keys())
    
    # Array for storing intensities, length = twice the number of frames
    intensity_profile = [0.0] * (2 * num_frames)

    # Keep track of last known coordinates
    last_coords = None
    
    for t in range(num_frames):
        if t < track_start:        
            # Before track start: use first known coords
            x_spot, y_spot = frame_to_spot[track_start]
        elif t > track_stop:
            # After track end: use last known coords
            x_spot, y_spot = frame_to_spot[track_stop]
        else:
            # Within track range: use spot if exists, else last known
            if t in frame_to_spot:
                x_spot, y_spot = frame_to_spot[t]
                last_coords = (x_spot, y_spot)
            else:
                x_spot, y_spot = last_coords
                print "There is no spot information for C%d track at frame %d -- Using coordinates from last available frame"%(channel, t)

        # Measure intensity
        mean_spot, mean_bg = measure(x_spot, y_spot, r, t+1, whole_FOV, BG)
        
        # Store the intensity data
        intensity_profile[num_frames + (t-track_stop)] += mean_spot
        
        # Write the results to the output file
        output.write("%d,%.4f,%.4f,%.4f,%.3f\n"%(t, x_spot, y_spot, mean_spot, mean_spot + mean_bg))
        
    return intensity_profile

###############################################

def open_image(path):
    imp = None
    try:
        imp = Opener.openUsingBioFormats(path)
        if imp is None:
            raise RuntimeError("BioFormats returned None")
    except Exception as e:
        print "BioFormats failed for %s: %s" % (path, e)
        print "Trying IJ.openImage()..."
        imp = IJ.openImage(path)

    if imp is None:
        raise RuntimeError("Could not open image: %s" % path)

    return imp

###############################################

def process(path_csv_colocalization, path_spots_C1, path_spots_C2, output_path):
    """
    Processes colocalization data from C1-C2 tracks and extracts intensity profiles
    for all events, using pre-built frame-to-spot dictionaries for fast access.
    """
    
    # Load colocalization CSV into ResultsTable
    IJ.open(path_csv_colocalization)
    filename_csv_coloc = os.path.basename(path_csv_colocalization)
    table_coloc = ResultsTable.getResultsTable(filename_csv_coloc)
    if table_coloc is None or table_coloc.size() == 0:
        raise ValueError("No data found in CSV file: " + filename_csv_coloc)
        
    column_COLOCALIZE_ID = table_coloc.getColumnAsDoubles(table_coloc.getColumnIndex("COLOCALIZE_ID"))
    colocalize_ids = set([int(x) for x in column_COLOCALIZE_ID])
        
    column_FILE_ID = table_coloc.getColumnAsDoubles(table_coloc.getColumnIndex("FILE_ID"))
    file_ids = set([int(x) for x in column_FILE_ID])
    
    print "\n*************************************"
    print "\tNumber of events:", len(colocalize_ids)
    print "\tNumber of movies:", len(file_ids)
    print "*************************************\n"
    
    # Group tracks by FILE_ID
    tracks_by_file = defaultdict(set)
    for i in range(table_coloc.size()):
        file_id = int(table_coloc.getStringValue("FILE_ID", i))
        coloc_id = int(table_coloc.getStringValue("COLOCALIZE_ID", i))
        tracks_by_file[file_id].add(coloc_id)
    
    # Initialize lists to store intensity data
    all_tracks_data_C1 = []
    all_tracks_data_C2 = []
    
    # Process each file ID
    for file_id, colocalize_ids in sorted(tracks_by_file.items()):

        print "\nProcessing FILE_ID=%d with %d events"%(file_id, len(colocalize_ids))
        filename_wholeFOV_C1 = experiment + "_" + extra_name_splitted_C1 + "_" + str(file_id) + ".tif"
        filename_wholeFOV_C2 = experiment + "_" + extra_name_splitted_C2 + "_" + str(file_id) + ".tif"
        path_C1 = os.path.join(path_movies_splitted, filename_wholeFOV_C1)
        path_C2 = os.path.join(path_movies_splitted, filename_wholeFOV_C2)
        
        print "FOV_C1:", filename_wholeFOV_C1
        print "FOV_C2:", filename_wholeFOV_C2
        print "Whole path C1:", path_C1
        print "Whole path C2:", path_C2
        
        whole_FOV_C1 = open_image(path_C1)
        whole_FOV_C2 = open_image(path_C2)
        
        num_channels = whole_FOV_C1.getNChannels()
        num_slices = whole_FOV_C1.getNSlices()
        num_frames = whole_FOV_C1.getNFrames()
        print "Channels: %d --- Slices (Z): %d --- Frames (T): %d"%(num_channels, num_slices, num_frames)

        # Process each track
        for colocalize_id in colocalize_ids:
            print "Colocalize_ID: %d"%(colocalize_id)
            
            # Get TRACK_ID for C1
            C1_track_id = None
            
            for i in range(table_coloc.size()):
                row_coloc_id = int(table_coloc.getStringValue("COLOCALIZE_ID", i))
                row_channel = int(table_coloc.getStringValue("CHANNEL", i))

                if row_coloc_id == colocalize_id and row_channel == 1:
                    C1_track_id = int(table_coloc.getStringValue("TRACK_ID", i))
                    break
                    
            if C1_track_id is None:
                raise ValueError("[ERROR] No TRACK_ID for COLOCALIZE_ID=%d in CHANNEL=1"%(coloc_id))

            # Path to spots CSVs
            filename_spots_C1 = experiment + "_C1_" + str(file_id) + "_spotsmodified.csv"
            filename_spots_C2 = "spots_modified_" + str(colocalize_id) + ".csv"
            path_allspots_C1 = os.path.join(path_spots_C1, filename_spots_C1)
            path_allspots_C2 = os.path.join(path_spots_C2, filename_spots_C2)
            
            if not os.path.exists(path_allspots_C1):
                raise IOError("[ERROR] Missing C1 spots file: " + path_allspots_C1)
            if not os.path.exists(path_allspots_C2):
                raise IOError("[ERROR] Missing C2 spots file: " + path_allspots_C2)
                
            # Output files
            filename_C1_intprofile = "Colocalized_ID_%d_C1.csv"%(colocalize_id)
            filename_C2_intprofile = "Colocalized_ID_%d_C2.csv"%(colocalize_id)
            outputC1 = open(os.path.join(output_path, filename_C1_intprofile), "w")
            outputC2 = open(os.path.join(output_path, filename_C2_intprofile), "w")
            outputC1.write("Frame,X,Y,Intensity_Corrected,Intensity_Raw\n")
            outputC2.write("Frame,X,Y,Intensity_Corrected,Intensity_Raw\n")
            
            # Get pre-built frame-to-spot dictionaries
            frame_to_spot_C1 = filter_allspots_C1(path_allspots_C1, C1_track_id)
            frame_to_spot_C2 = process_spots_C2(path_allspots_C2)
            
            if not frame_to_spot_C1:
                raise ValueError("[ERROR] No C1 spots found for TRACK_ID=%d in file: %s" % (C1_track_id, path_allspots_C1))
            if not frame_to_spot_C2:
                raise ValueError("[ERROR] No C2 spots found for COLOCALIZE_ID=%d in file: %s" % (colocalize_id, path_allspots_C2))
            
            # Extract intensity profiles
            int_profile_C1 = extract_intensity_profile(
                whole_FOV = whole_FOV_C1, frame_to_spot = frame_to_spot_C1, num_frames = num_frames, output = outputC1, channel = 1
            )
            int_profile_C2 = extract_intensity_profile(
                whole_FOV = whole_FOV_C2, frame_to_spot = frame_to_spot_C2, num_frames = num_frames, output = outputC2, channel = 2
                )
            
            all_tracks_data_C1.append(int_profile_C1)
            all_tracks_data_C2.append(int_profile_C2)
            
            outputC1.close()
            outputC2.close()
    
    return all_tracks_data_C1, all_tracks_data_C2, num_frames


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
    # Retrieve stored preferences or use built-in defaults
    
    DEFAULTS = {
        "experiment": "",
        "extra_name_splitted_C1": "splitted_C1",
        "extra_name_splitted_C2": "splitted_C2",
        "root": IJ.getDirectory("home"),
    }
    
    default_experiment              = prefs.get(None, "experiment", DEFAULTS["experiment"])
    default_extra_name_splitted_C1  = prefs.get(None, "extra_name_splitted_C1", DEFAULTS["extra_name_splitted_C1"])
    default_extra_name_splitted_C2  = prefs.get(None, "extra_name_splitted_C2", DEFAULTS["extra_name_splitted_C2"])
    default_root                = prefs.get(None, "root_folder", DEFAULTS["root"])
    default_csv_colocalization  = prefs.get(None, "csv_colocalization", "")
    default_plot_in_ImageJ      = bool(prefs.get(None, "plot_in_ImageJ", ""))

    # Create the dialog
    gui_initial = GenericDialogPlus("Input from user")
    
    # Basic information fields
    gui_initial.addStringField("Experiment name:", default_experiment)
    gui_initial.addToSameRow()
    gui_initial.addMessage("The input videos must match the following convention:\n   ExperimentName_ExtraNameFOV_FileID.tif")
    gui_initial.addDirectoryField("Root folder:", default_root, 40)
    gui_initial.addFileField("CSV file (C1-C2 colocalizing):", default_csv_colocalization, 40)
    
    # Naming parameters with descriptions
    gui_initial.addMessage("Naming Parameters:")
    gui_initial.addStringField("Extra name FOV C1", default_extra_name_splitted_C1)
    gui_initial.addToSameRow()
    gui_initial.addMessage("(Text after experiment name in input video names. Usually splitted_C1)")
    gui_initial.addStringField("Extra name FOV C2", default_extra_name_splitted_C2)
    gui_initial.addToSameRow()
    gui_initial.addMessage("(Text after experiment name in input video names. Usually splitted_C2)")
    
    gui_initial.addCheckbox("Display 5 intensity profiles as example in ImageJ", default_plot_in_ImageJ)

    gui_initial.showDialog()

    if gui_initial.wasOKed():
        # Retrieve user inputs in order of appearance
        experiment          = gui_initial.getNextString()
        root_folder         = gui_initial.getNextString()
        csv_colocalization  = gui_initial.getNextString()

        extra_name_splitted_C1       = gui_initial.getNextString()
        extra_name_splitted_C2       = gui_initial.getNextString()
        
        plot_in_ImageJ   = gui_initial.getNextBoolean()
        # Save the new defaults for future sessions
        
        prefs_dict = {
            "experiment": experiment,
            "extra_name_splitted_C1": extra_name_splitted_C1,
            "extra_name_splitted_C2": extra_name_splitted_C2,
            "root_folder": root_folder,
            "csv_colocalization": csv_colocalization,
            "plot_in_ImageJ": plot_in_ImageJ,
        }
        
        for key, value in prefs_dict.items():
            prefs.put(None, key, value)

        # Process paths: Create sub-folders for movies and CSVs.
        root_folder = os.path.normpath(root_folder)
        
        print "\n*************************************"
        print "\tRoot folder:", root_folder
        print "*************************************\n"

        path_movies_splitted    = os.path.join(root_folder, "2_Splitted")
        path_spots_C1           = os.path.join(root_folder, "4_Tracking", "individual_movies_spots_modified")
        path_spots_C2           = os.path.join(root_folder, "5_Analysis", "Spots_C2")
        path_intensity_profiles = os.path.join(root_folder, "5_Analysis", "Intensity_Profiles")

        if not os.path.exists(root_folder):
            raise Exception("\n\n\n\nError: The root path does not exist: {0}\n\n\n\n".format(root_folder))
        if not os.path.exists(path_movies_splitted):
            raise Exception("\n\n\n\nError: The path for splitted movies does not exist: {0}\n\n\n\n".format(path_movies_splitted))
        if not os.path.exists(path_spots_C1):
            raise Exception("\n\n\n\nError: The path for C1 spots information does not exist: {0}\n\n\n\n".format(path_spots_C1))
        if not os.path.exists(path_spots_C2):
            raise Exception("\n\n\n\nError: The path for C2 spots information does not exist: {0}\n\n\n\n".format(path_spots_C2))
        if not os.path.exists(path_intensity_profiles):
            os.makedirs(path_intensity_profiles)
        
        params = {
            "experiment": experiment,
            "csv_colocalization": csv_colocalization,
            "extra_name_splitted_C1": extra_name_splitted_C1,
            "extra_name_splitted_C2": extra_name_splitted_C2,
            "path_movies_splitted": path_movies_splitted,
            "path_spots_C1": path_spots_C1,
            "path_spots_C2": path_spots_C2,
            "path_intensity_profiles": path_intensity_profiles,
            "plot_in_ImageJ": plot_in_ImageJ,
        }
        
        return params
    
    else:
        raise Exception("Initial GUI was cancelled.")


################################################################
######################## RUN EVERYTHING ########################
################################################################

params = show_gui_initial()

experiment = params["experiment"]
csv_colocalization = params["csv_colocalization"]
extra_name_splitted_C1 = params["extra_name_splitted_C1"]
extra_name_splitted_C2 = params["extra_name_splitted_C2"]
path_movies_splitted = params["path_movies_splitted"]
path_spots_C1 = params["path_spots_C1"]
path_spots_C2 = params["path_spots_C2"]
path_intensity_profiles = params["path_intensity_profiles"]
plot_in_ImageJ = params["plot_in_ImageJ"]


data_C1, data_C2, num_frames = process(
    path_csv_colocalization = csv_colocalization, 
    path_spots_C1 = path_spots_C1,
    path_spots_C2 = path_spots_C2,
    output_path = path_intensity_profiles,
)
                     

if plot_in_ImageJ:
    
    MAX_PLOTS = 5
    
    def moving_average(data, window_size=5):
        smoothed = []
        half_window = window_size // 2
        for i in range(len(data)):
            window = data[max(0, i - half_window):min(len(data), i + half_window + 1)]
            smoothed.append(sum(window) / float(len(window)))
        return smoothed
        
    xx = [t*0.7 for t in range(-num_frames,num_frames)]
    
    for i in range(min(len(data_C1), MAX_PLOTS)):
        norm_C1 = [x / max(data_C1[i]) for x in data_C1[i]]
        norm_C2 = [x / max(data_C2[i]) for x in data_C2[i]]
        smooth_C1 = moving_average(norm_C1, window_size=10)
        smooth_C2 = moving_average(norm_C2, window_size=10)
        
        plot = Plot("Track %d" % i, "Time", "Intensity")
        plot.addPoints(xx, smooth_C1, Plot.LINE)
        plot.setColor("green")
        plot.addPoints(xx, smooth_C2, Plot.LINE)
        plot.setColor("red")
        
        plot.show()


