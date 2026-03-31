import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

#############################################################################################################
############################################# HELPER FUNCTIONS ##############################################
#############################################################################################################

def calculate_time_shift(zero_at, track_start_C1, track_stop_C1, track_start_C2, track_stop_C2):
    """
    Calculates the time shift required to align intensity profiles.
    """
    shift_options = {
        "C1_center": (track_start_C1 + track_stop_C1) / 2,
        "C1_start":  track_start_C1,
        "C1_stop":   track_stop_C1,
        "C2_center": (track_start_C2 + track_stop_C2) / 2,
        "C2_start":  track_start_C2,
        "C2_stop":   track_stop_C2,
    }
    return shift_options.get(zero_at)


def calculate_track_positions(t_min, t_max, time_shift, track_start_C1, track_stop_C1,
                              track_start_C2, track_stop_C2, zero_at, normalize_duration):
    """
    Calculates adjusted track start and stop positions, considering time shift and normalization.
    """
    track_start_C1_shifted = track_start_C1 - time_shift
    track_stop_C1_shifted  = track_stop_C1  - time_shift
    track_start_C2_shifted = track_start_C2 - time_shift
    track_stop_C2_shifted  = track_stop_C2  - time_shift

    if not normalize_duration:
        return track_start_C1_shifted, track_stop_C1_shifted, track_start_C2_shifted, track_stop_C2_shifted

    duration = t_max - t_min
    t_normalized_start_C1 = (track_start_C1_shifted - t_min) / duration
    t_normalized_stop_C1  = (track_stop_C1_shifted  - t_min) / duration
    t_normalized_start_C2 = (track_start_C2_shifted - t_min) / duration
    t_normalized_stop_C2  = (track_stop_C2_shifted  - t_min) / duration

    zero_position_normalized = (0 - t_min) / duration

    track_start_C1_adjusted = t_normalized_start_C1 - zero_position_normalized
    track_stop_C1_adjusted  = t_normalized_stop_C1  - zero_position_normalized
    track_start_C2_adjusted = t_normalized_start_C2 - zero_position_normalized
    track_stop_C2_adjusted  = t_normalized_stop_C2  - zero_position_normalized

    return track_start_C1_adjusted, track_stop_C1_adjusted, track_start_C2_adjusted, track_stop_C2_adjusted

#############################################################################################################
############################################ INDIVIDUAL PROFILES ############################################
#############################################################################################################

def load_intensity_from_memory(intensity_files, channel, colocalize_id):
    """
    Reads an intensity profile CSV from the in-memory dict.
    Replaces the original filesystem-based load_intensity_data().

    Parameters:
    - intensity_files: dict {filename: bytes}  — populated from the uploaded zip
    - channel: int (1 or 2)
    - colocalize_id: int

    Returns:
    - pd.DataFrame or None if file is missing / unreadable
    """
    filename = f"Colocalized_ID_{colocalize_id}_C{channel}.csv"
    if filename not in intensity_files:
        logging.warning(f"File not found in memory: {filename}")
        return None
    try:
        return pd.read_csv(io.BytesIO(intensity_files[filename]))
    except Exception as e:
        logging.error(f"Failed to parse {filename}: {e}")
        return None


def data_individual_profiles(df_colocalization, intensity_files, config):
    """
    Extract and normalize intensity profiles from colocalization data.

    Parameters:
    - df_colocalization: pd.DataFrame
    - intensity_files: dict {filename: bytes}
        In-memory file store built from the uploaded zip.
        Replaces the original `path_intensity_profiles` folder path.
    - config: dict

    Returns:
    - dict with processed intensities and metadata for C1 and C2.
    """
    list_colocalize_IDs = df_colocalization.COLOCALIZE_ID.unique()

    all_intensities_C1, all_intensities_C2 = [], []
    all_timepoints, all_timeshifts = [], []
    all_info_track_C1, all_info_track_C2 = [], []
    global_min_time, global_max_time = np.inf, -np.inf
    adjusted_positions = {"C1_start": [], "C1_stop": [], "C2_start": [], "C2_stop": []}

    for colocalize_id in list_colocalize_IDs:
        info_track_C1 = df_colocalization.query("COLOCALIZE_ID == @colocalize_id & CHANNEL == 1")
        info_track_C2 = df_colocalization.query("COLOCALIZE_ID == @colocalize_id & CHANNEL == 2")

        if info_track_C1.empty or info_track_C2.empty:
            logging.warning(f"No data found for Colocalize ID {colocalize_id}. Skipping...")
            continue

        track_start_C1 = int(info_track_C1["TRACK_START"].iloc[0])
        track_stop_C1  = int(info_track_C1["TRACK_STOP"].iloc[0])
        track_start_C2 = int(info_track_C2["TRACK_START"].iloc[0])
        track_stop_C2  = int(info_track_C2["TRACK_STOP"].iloc[0])

        # ── Key change: use in-memory loader instead of os.path.join ──────
        df_C1 = load_intensity_from_memory(intensity_files, 1, colocalize_id)
        df_C2 = load_intensity_from_memory(intensity_files, 2, colocalize_id)
        if df_C1 is None or df_C2 is None:
            continue

        time_shift = calculate_time_shift(
            config["zero_at"], track_start_C1, track_stop_C1, track_start_C2, track_stop_C2
        )
        all_timeshifts.append(time_shift)

        timepoints_shifted = np.arange(len(df_C1)) - time_shift

        intensity_C1_wholemovie = df_C1["Intensity_Corrected"]
        intensity_C2_wholemovie = df_C2["Intensity_Corrected"]

        if config["normalize_intensity"]:
            intensity_track_median_C1 = np.median(intensity_C1_wholemovie[track_start_C1:track_stop_C1])
            intensity_track_median_C2 = np.median(intensity_C2_wholemovie[track_start_C2:track_stop_C2])
            intensity_C1_wholemovie = intensity_C1_wholemovie / intensity_track_median_C1
            intensity_C2_wholemovie = intensity_C2_wholemovie / intensity_track_median_C2

        if config["normalize_duration"]:
            t_min = track_start_C2 if config["normalize_by_C2"] else track_start_C1
            t_max = track_stop_C2  if config["normalize_by_C2"] else track_stop_C1
            t_normalized = timepoints_shifted / (t_max - t_min)

            interp_t_individual = np.linspace(t_normalized.min(), t_normalized.max(), len(df_C1))
            interp_C1 = interp1d(t_normalized, intensity_C1_wholemovie, bounds_error=False, fill_value="extrapolate")
            interp_C2 = interp1d(t_normalized, intensity_C2_wholemovie, bounds_error=False, fill_value="extrapolate")

            all_intensities_C1.append(interp_C1(interp_t_individual))
            all_intensities_C2.append(interp_C2(interp_t_individual))
            all_timepoints.append(interp_t_individual)

            global_min_time = min(global_min_time, interp_t_individual.min())
            global_max_time = max(global_max_time, interp_t_individual.max())

        else:
            all_intensities_C1.append(intensity_C1_wholemovie)
            all_intensities_C2.append(intensity_C2_wholemovie)
            all_timepoints.append(timepoints_shifted)

        all_info_track_C1.append(info_track_C1)
        all_info_track_C2.append(info_track_C2)

        # Calculate adjusted track positions
        if config["duration_in_seconds"]:
            fr = config["framerate"]
            ts1, te1 = track_start_C1 * fr, track_stop_C1 * fr
            ts2, te2 = track_start_C2 * fr, track_stop_C2 * fr
            tshift_s = time_shift * fr
            t_min = te2 if config["normalize_by_C2"] else ts1
            t_max = te2 if config["normalize_by_C2"] else te1
            adjusted_pos = calculate_track_positions(
                t_min, t_max, tshift_s, ts1, te1, ts2, te2,
                config["zero_at"], config["normalize_duration"]
            )
        else:
            t_min = track_start_C2 if config["normalize_by_C2"] else track_start_C1
            t_max = track_stop_C2  if config["normalize_by_C2"] else track_stop_C1
            adjusted_pos = calculate_track_positions(
                t_min, t_max, time_shift,
                track_start_C1, track_stop_C1, track_start_C2, track_stop_C2,
                config["zero_at"], config["normalize_duration"]
            )

        adjusted_positions["C1_start"].append(adjusted_pos[0])
        adjusted_positions["C1_stop"].append(adjusted_pos[1])
        adjusted_positions["C2_start"].append(adjusted_pos[2])
        adjusted_positions["C2_stop"].append(adjusted_pos[3])

    return {
        "intensities_C1":    all_intensities_C1,
        "intensities_C2":    all_intensities_C2,
        "timepoints":        all_timepoints,
        "timeshifts":        all_timeshifts,
        "info_track_C1":     all_info_track_C1,
        "info_track_C2":     all_info_track_C2,
        "global_min_exp":    global_min_time,
        "global_max_exp":    global_max_time,
        "adjusted_positions": adjusted_positions,
    }

#############################################################################################################
#############################################################################################################
#############################################################################################################

def plot_individual_profiles(data_individual_profiles, config):
    """
    Plot intensity profiles for colocalized tracks.
    Returns the matplotlib Figure so Streamlit can render it with st.pyplot().
    """
    all_intensities_C1  = data_individual_profiles["intensities_C1"]
    all_intensities_C2  = data_individual_profiles["intensities_C2"]
    all_timepoints      = data_individual_profiles["timepoints"]
    all_timeshifts      = data_individual_profiles["timeshifts"]
    all_info_track_C1   = data_individual_profiles["info_track_C1"]
    all_info_track_C2   = data_individual_profiles["info_track_C2"]
    adjusted_positions  = data_individual_profiles["adjusted_positions"]

    num_tracks = len(all_intensities_C1)
    rows = (num_tracks + config["columns_individual"] - 1) // config["columns_individual"]

    fig, axes = plt.subplots(
        rows, config["columns_individual"],
        figsize=(config["figsize_individual"][0] * config["columns_individual"],
                 config["figsize_individual"][1] * rows),
        dpi=config.get("dpi", 100),
        sharex=True, sharey=True,
    )
    axes = np.ravel(axes)

    assert len(all_intensities_C1) == len(all_intensities_C2) == len(all_timepoints), \
        "Data dimensions must match!"

    for i in range(num_tracks):
        intensities_C1 = all_intensities_C1[i]
        intensities_C2 = all_intensities_C2[i]
        timepoints     = all_timepoints[i]
        time_shift     = all_timeshifts[i]
        info_track_C1  = all_info_track_C1[i]
        info_track_C2  = all_info_track_C2[i]

        track_start_C1 = int(info_track_C1["TRACK_START"].iloc[0])
        track_stop_C1  = int(info_track_C1["TRACK_STOP"].iloc[0])
        track_start_C2 = int(info_track_C2["TRACK_START"].iloc[0])
        track_stop_C2  = int(info_track_C2["TRACK_STOP"].iloc[0])

        min_relevant = min(track_start_C1, track_stop_C1, track_start_C2, track_stop_C2)
        max_relevant = max(track_start_C1, track_stop_C1, track_start_C2, track_stop_C2)
        start_slice  = max(0, min_relevant - config["extra_frames"])
        end_slice    = min(len(intensities_C1), max_relevant + config["extra_frames"])

        intensities_C1_track = intensities_C1[start_slice:end_slice]
        intensities_C2_track = intensities_C2[start_slice:end_slice]
        timepoints_track     = timepoints[start_slice:end_slice]

        if config["duration_in_seconds"] and not config["normalize_duration"]:
            timepoints_track = timepoints_track * config["framerate"]

        if config["smooth_intensity"]:
            intensities_C1_track = gaussian_filter1d(intensities_C1_track, sigma=config["smooth_sigma"])
            intensities_C2_track = gaussian_filter1d(intensities_C2_track, sigma=config["smooth_sigma"])

        axes[i].plot(timepoints_track, intensities_C1_track, label="C1", color="red",   alpha=0.8)
        axes[i].plot(timepoints_track, intensities_C2_track, label="C2", color="green", alpha=0.8)
        axes[i].legend(loc="upper left", fontsize=8)

        if config["plot_titles"]:
            fov           = int(info_track_C1["FILE_ID"].iloc[0])
            colocalize_id = int(info_track_C1["COLOCALIZE_ID"].iloc[0])
            axes[i].set_title(
                f"FOV: {fov} - ID: {colocalize_id}\n"
                f"C1: {track_start_C1}-{track_stop_C1}\n"
                f"C2: {track_start_C2}-{track_stop_C2}",
                fontsize=10,
            )

        if config["plot_duration"]:
            dur_C1 = int(info_track_C1["TRACK_DURATION"].iloc[0])
            dur_C2 = int(info_track_C2["TRACK_DURATION"].iloc[0])
            if config["duration_in_seconds"]:
                txt = f"Dur C1: {dur_C1 * config['framerate']:.2f} s\nDur C2: {dur_C2 * config['framerate']:.2f} s"
            else:
                txt = f"Dur C1: {dur_C1}\nDur C2: {dur_C2}"
            axes[i].text(0.5, 0.1, txt, ha="center", transform=axes[i].transAxes, fontsize=8)

        if config["plot_thresholds"]:
            axes[i].axvline(x=adjusted_positions["C1_start"][i], color="red",   linestyle="--", alpha=0.5)
            axes[i].axvline(x=adjusted_positions["C1_stop"][i],  color="red",   linestyle="--", alpha=0.5)
            axes[i].axvline(x=adjusted_positions["C2_start"][i], color="green", linestyle="--", alpha=0.5)
            axes[i].axvline(x=adjusted_positions["C2_stop"][i],  color="green", linestyle="--", alpha=0.5)

    for i in range(num_tracks, len(axes)):
        fig.delaxes(axes[i])

    xlabel = "Normalized duration" if config["normalize_duration"] else "Duration (s)"
    ylabel = "Normalized intensity" if config["normalize_intensity"] else "Intensity (A.U.)"
    fig.supxlabel(xlabel, fontsize=16)
    fig.supylabel(ylabel, fontsize=16)
    fig.suptitle(config.get("title_figure", ""), fontsize=20)

    plt.tight_layout()

    return fig

#############################################################################################################
############################################# AVERAGE PROFILES ##############################################
#############################################################################################################

def interpolate_and_average(data, global_timepoints, config):
    """Interpolates and averages intensity profiles."""
    if not data["intensities"]:
        return None

    interpolated_profiles = []
    for timepoint, intensity in zip(data["timepoints"], data["intensities"]):
        f = interp1d(timepoint, intensity, bounds_error=False, fill_value=np.nan)
        interpolated_profiles.append(f(global_timepoints))
    interpolated_profiles = np.array(interpolated_profiles)

    avg_intensity = np.nanmean(interpolated_profiles, axis=0)

    std = np.nanstd(interpolated_profiles, axis=0)
    # N per timepoint
    n = np.sum(~np.isnan(interpolated_profiles), axis=0)
    # 95% CI half-width
    CI_95 = 1.96 * std / np.sqrt(n)
    
    if config["plot_intensity_error"] == "Standard deviation":
        error = std
    elif config["plot_intensity_error"] == "Confidence interval (95%)":
        error = CI_95
    else:
    	error = std

    if config["smooth_intensity"]:
        avg_intensity = gaussian_filter1d(avg_intensity, sigma=config["smooth_sigma"])
        error         = gaussian_filter1d(error,         sigma=config["smooth_sigma"])

    return {"values": avg_intensity, "error": error}


def compute_statistics(values):
    """Compute mean, median, std and 95% CI, handling empty lists."""
    if not values:
        return {"mean": None, "median": None, "std": None, "ci": None}
    return {
        "mean":   np.mean(values),
        "median": np.median(values),
        "std":    np.std(values),
        "ci":     bootstrap_CI(values, n_bootstraps=1000),
    }


def bootstrap_CI(data, n_bootstraps=1000):
    """95% confidence interval for the median via bootstrapping."""
    bootstrap_medians = [
        np.median(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstraps)
    ]
    bootstrap_medians = np.array(bootstrap_medians)
    return (np.percentile(bootstrap_medians, 2.5), np.percentile(bootstrap_medians, 97.5))


def compute_average_profile(data_individual_profiles_list, protein_names_C2, config):
    """
    Computes average intensity profiles for C1 and C2 proteins.
    Unchanged from original — operates on already-processed data dicts.
    """
    data = {}
    data[config["C1_name"]] = {
        "all": {"intensities": [], "timepoints": [], "start": [], "stop": []},
        "per_experiment": {},
    }
    for protein_name in protein_names_C2:
        data[protein_name] = {"intensities": [], "timepoints": [], "start": [], "stop": []}

    global_min_time_raw  = float("inf")
    global_max_time_raw  = float("-inf")
    global_min_time_norm_list = []
    global_max_time_norm_list = []
    min_time_interval = float("inf")

    for experiment_data, protein_name in zip(data_individual_profiles_list, protein_names_C2):
        intensities_C1    = experiment_data["intensities_C1"]
        intensities_C2    = experiment_data["intensities_C2"]
        raw_timepoints    = experiment_data["timepoints"]
        global_min_exp    = experiment_data["global_min_exp"]
        global_max_exp    = experiment_data["global_max_exp"]
        adjusted_positions = experiment_data["adjusted_positions"]

        if (not config.get("normalize_duration")
                and config.get("duration_in_seconds", False)
                and config.get("framerate")):
            timepoints = [np.array(tp) * config["framerate"] for tp in raw_timepoints]
        else:
            timepoints = raw_timepoints

        data[config["C1_name"]]["all"]["intensities"].extend(intensities_C1)
        data[config["C1_name"]]["all"]["timepoints"].extend(timepoints)
        data[protein_name]["intensities"].extend(intensities_C2)
        data[protein_name]["timepoints"].extend(timepoints)

        data[config["C1_name"]]["all"]["start"].extend(adjusted_positions["C1_start"])
        data[config["C1_name"]]["all"]["stop"].extend(adjusted_positions["C1_stop"])
        data[protein_name]["start"].extend(adjusted_positions["C2_start"])
        data[protein_name]["stop"].extend(adjusted_positions["C2_stop"])

        data[config["C1_name"]]["per_experiment"][protein_name] = {
            "intensities": intensities_C1,
            "timepoints":  timepoints,
            "start":       adjusted_positions["C1_start"],
            "stop":        adjusted_positions["C1_stop"],
        }

        global_min_time_raw = min(global_min_time_raw, min(min(tp) for tp in timepoints))
        global_max_time_raw = max(global_max_time_raw, max(max(tp) for tp in timepoints))
        global_min_time_norm_list.append(global_min_exp)
        global_max_time_norm_list.append(global_max_exp)

        time_intervals = np.diff(raw_timepoints)
        min_time_interval = min(min_time_interval, np.min(time_intervals))

    if config["normalize_duration"]:
        global_min_time_norm = np.min(global_min_time_norm_list)
        global_max_time_norm = np.max(global_max_time_norm_list)
        total_duration = global_max_time_norm - global_min_time_norm
        num_points = int(np.ceil(total_duration / min_time_interval))
        global_timepoints = np.linspace(global_min_time_norm, global_max_time_norm, num_points)
    else:
        step = config["framerate"] if config.get("framerate") else 1
        global_timepoints = np.arange(global_min_time_raw, global_max_time_raw + 1, step)

    computed = {
        "global_timepoints": global_timepoints,
        config["C1_name"]: {"all": {}, "per_experiment": {}},
    }

    for protein_name in data.keys():
        if protein_name == config["C1_name"]:
            data_all = data[protein_name]["all"]
            computed[protein_name]["all"] = {
                "avg_intensity": interpolate_and_average(data_all, global_timepoints, config),
                "num_profiles":  len(data_all["intensities"]),
                "start":         compute_statistics(data_all["start"]),
                "stop":          compute_statistics(data_all["stop"]),
            }
            for exp_protein, exp_data in data[protein_name]["per_experiment"].items():
                computed[protein_name]["per_experiment"][exp_protein] = {
                    "avg_intensity": interpolate_and_average(exp_data, global_timepoints, config),
                    "num_profiles":  len(exp_data["intensities"]),
                    "start":         compute_statistics(exp_data["start"]),
                    "stop":          compute_statistics(exp_data["stop"]),
                }
        else:
            computed[protein_name] = {
                "avg_intensity": interpolate_and_average(data[protein_name], global_timepoints, config),
                "num_profiles":  len(data[protein_name]["intensities"]),
                "start":         compute_statistics(data[protein_name]["start"]),
                "stop":          compute_statistics(data[protein_name]["stop"]),
            }

    return computed, data


def plot_average_profile(computed, config):
    """
    Plot average intensity profiles.
    Returns the matplotlib Figure so Streamlit can render it with st.pyplot().
    """
    assert len(config["protein_names"]) == len(config["colors"]), \
        "Number of colors does not match number of proteins"

    global_timepoints = computed["global_timepoints"]
    fig, ax = plt.subplots(figsize=config["figsize"], dpi=config["dpi"])

    y_min, y_max = config["ylim"]
    y_range      = y_max - y_min
    offset       = y_range * 0.02
    marker_y_base = y_min + offset

    # When split_C1 is on, we expand C1 into per-experiment entries before
    # iterating, so the rest of the loop stays identical for every trace.
    plot_entries = []  # list of (label, color, protein_data)
    
    for i, protein_name in enumerate(config["protein_names"]):
        color = config["colors"][i]

        is_C1 = (protein_name == config["C1_name"])
        
        if is_C1 and config.get("split_C1_by_experiment"):
            # Generate N red shades, one per experiment
            per_exp = computed[protein_name]["per_experiment"]
            red_shades = _red_shades(len(per_exp))
            for shade, (exp_protein, exp_data) in zip(red_shades, per_exp.items()):
                label = f"{protein_name} ({exp_protein})"
                plot_entries.append((label, shade, exp_data))
        elif is_C1:
            protein_data = computed[protein_name]["all"]
            plot_entries.append((protein_name, color, protein_data))
        else:
            plot_entries.append((protein_name, color, computed[protein_name]))
            
    for i, (label, color, protein_data) in enumerate(plot_entries):
        avg_intensity = protein_data["avg_intensity"]["values"]
        error         = protein_data["avg_intensity"]["error"]

        ax.plot(
            global_timepoints, avg_intensity,
            label=f"{label} (N={protein_data['num_profiles']})",
            color=color, linewidth=2, alpha=0.8,
        )

        if config["plot_intensity_error"] != "None":
            ax.fill_between(
                global_timepoints,
                avg_intensity - error,
                avg_intensity + error,
                color=color, alpha=0.2,
            )

        if config["plot_start_stop_central"]:
            start = protein_data["start"][config["type_start_stop_central"]]
            stop  = protein_data["stop"][config["type_start_stop_central"]]
            ax.axvline(x=start, color=color, linestyle="--", alpha=config["alpha_start_stop_central"])
            ax.axvline(x=stop,  color=color, linestyle="--", alpha=config["alpha_start_stop_central"])

            if config["plot_start_stop_error"]:
                if config["type_start_stop_error"] == "ci":
                    s_lo, s_hi = protein_data["start"]["ci"]
                    e_lo, e_hi = protein_data["stop"]["ci"]
                    xerr_start = [[abs(start - s_lo)], [abs(s_hi - start)]]
                    xerr_stop  = [[abs(stop  - e_lo)], [abs(e_hi - stop)]]
                else:  # std
                    xerr_start = protein_data["start"]["std"]
                    xerr_stop  = protein_data["stop"]["std"]

                marker_y = marker_y_base + i * offset
                ax.errorbar(start, marker_y, xerr=xerr_start, fmt="|",
                            color=color, capsize=5, alpha=config["alpha_start_stop_error"])
                ax.errorbar(stop,  marker_y, xerr=xerr_stop,  fmt="|",
                            color=color, capsize=5, alpha=config["alpha_start_stop_error"])

    if config.get("xticks") is not None:
        ax.xaxis.set_major_locator(MultipleLocator(config["xticks"]))

    if config["normalize_duration"]:
        xlabel = "Normalized Time"
    else:
        xlabel = "Time (s)" if config.get("duration_in_seconds", False) else "Time (frames)"

    ax.set(
        xlabel=xlabel,
        ylabel="Normalized Intensity" if config.get("normalize_intensity", False) else "Intensity (AU)",
        xlim=config.get("xlim"),
        ylim=config.get("ylim"),
    )
    ax.set_title(config.get("title_average", "Average Intensity Profile"))
    ax.legend(fontsize=config.get("fontsize_legend", 10))
    sns.despine(left=False, bottom=False)

    plt.tight_layout()

    # ── Key change: return fig instead of calling plt.show() ──────────────
    return fig


def process_experiments(loaded_experiments, config):
    """
    Processes multiple experiments and returns computed average profile data.

    Parameters:
    - loaded_experiments: list of ExperimentData objects (from data_loader.py)
    - config: dict

    Returns:
    - averaged_data: dict  (pass to plot_average_profile)
    - individual_data: dict
    """
    protein_names_C2 = [exp.protein_name for exp in loaded_experiments]
    config["protein_names"] = [config["C1_name"]] + protein_names_C2

    data_individual_profiles_list = []
    for exp in loaded_experiments:
        data_i = data_individual_profiles(
            df_colocalization=exp.df_colocalization,
            intensity_files=exp.intensity_files,
            config=config,
        )
        data_individual_profiles_list.append(data_i)

    averaged_data, individual_data = compute_average_profile(
        data_individual_profiles_list=data_individual_profiles_list,
        protein_names_C2=protein_names_C2,
        config=config,
    )

    return averaged_data, individual_data

def _red_shades(n):
    """
    Returns n evenly spaced hex colors between a light and dark red.
    With n=1 returns the canonical mid-red used for aggregated C1.
    """
    import matplotlib.colors as mcolors
    light = np.array(mcolors.to_rgb("#f1948a"))  # light red
    dark  = np.array(mcolors.to_rgb("#922b21"))  # dark red
    if n == 1:
        return ["#e74c3c"]  # same as DEFAULT_COLORS[0], consistent with aggregated mode
    return [
        "#{:02x}{:02x}{:02x}".format(
            int(c[0]*255), int(c[1]*255), int(c[2]*255)
        )
        for c in [light + t * (dark - light) for t in np.linspace(0, 1, n)]
    ]

#############################################################################################################
################################################ TIMELINES ##################################################
#############################################################################################################

def align_bars(df, zero_at="C1_start"):
    """
    Aligns track start/stop times to a reference point for each COLOCALIZE_ID.

    Parameters:
    - df: DataFrame with columns [COLOCALIZE_ID, CHANNEL, TRACK_START, TRACK_STOP]
    - zero_at: one of C1_start, C1_stop, C1_center, C2_start, C2_stop, C2_center

    Returns:
    - aligned_df with ADJUSTED_START, ADJUSTED_STOP, and POSITION columns added.
    """
    aligned_df = df.copy()
    aligned_df["ADJUSTED_START"] = np.nan
    aligned_df["ADJUSTED_STOP"]  = np.nan

    for coloc_id in df["COLOCALIZE_ID"].unique():
        subset = df[df["COLOCALIZE_ID"] == coloc_id]
        c1 = subset[subset["CHANNEL"] == 1]
        c2 = subset[subset["CHANNEL"] == 2]

        if zero_at == "C1_start":
            ref_time = c1["TRACK_START"].values[0]
        elif zero_at == "C1_stop":
            ref_time = c1["TRACK_STOP"].values[0]
        elif zero_at == "C1_center":
            ref_time = (c1["TRACK_START"].values[0] + c1["TRACK_STOP"].values[0]) / 2
        elif zero_at == "C2_start":
            ref_time = c2["TRACK_START"].values[0]
        elif zero_at == "C2_stop":
            ref_time = c2["TRACK_STOP"].values[0]
        elif zero_at == "C2_center":
            ref_time = (c2["TRACK_START"].values[0] + c2["TRACK_STOP"].values[0]) / 2
        else:
            raise ValueError(f"Invalid zero_at value: {zero_at}")

        aligned_df.loc[df["COLOCALIZE_ID"] == coloc_id, "ADJUSTED_START"] = (
            subset["TRACK_START"] - ref_time
        )
        aligned_df.loc[df["COLOCALIZE_ID"] == coloc_id, "ADJUSTED_STOP"] = (
            subset["TRACK_STOP"] - ref_time
        )

    unique_ids = aligned_df["COLOCALIZE_ID"].unique()
    id_to_pos  = {cid: idx for idx, cid in enumerate(unique_ids)}
    aligned_df["POSITION"] = aligned_df["COLOCALIZE_ID"].map(id_to_pos)

    return aligned_df


def sort_by_duration(df, channel_to_sort=1, ascending=True):
    """
    Sorts tracks by duration of the specified channel, keeping paired rows together.
    """
    if channel_to_sort not in df["CHANNEL"].unique():
        raise ValueError(f"Channel {channel_to_sort} not found in DataFrame.")

    df_sorted_ch  = df[df["CHANNEL"] == channel_to_sort].sort_values(
        by="TRACK_DURATION", ascending=ascending
    )
    df_others = df[df["CHANNEL"] != channel_to_sort]
    return pd.concat([df_sorted_ch, df_others], ignore_index=True)


def plot_individual_timelines(df_colocalization, config):
    """
    Plots one horizontal bar per track pair for a single experiment.

    Parameters:
    - df_colocalization: pd.DataFrame  (raw, not yet aligned)
    - config: dict with keys:
        zero_at, tl_zero_at, duration_in_seconds, framerate,
        tl_sort_channel, tl_sort_ascending,
        tl_colorC1, tl_colorC2, tl_xlim, tl_xticks,
        tl_figsize, dpi, tl_title_individual,
        C1_name  (used as C2 label fallback if protein_name not available)

    Returns:
    - fig: matplotlib Figure
    """
    df_sorted  = sort_by_duration(
        df_colocalization,
        channel_to_sort=config["tl_sort_channel"],
        ascending=config["tl_sort_ascending"],
    )
    df_aligned = align_bars(df_sorted, zero_at=config["tl_zero_at"])

    # Framerate conversion
    if config["duration_in_seconds"] and config.get("framerate"):
        fr = config["framerate"]
        df_aligned["ADJUSTED_START"]  = df_aligned["ADJUSTED_START"]  * fr
        df_aligned["ADJUSTED_STOP"]   = df_aligned["ADJUSTED_STOP"]   * fr
        df_aligned["TRACK_DURATION"]  = df_aligned["TRACK_DURATION"]  * fr
        xlabel = "Time (s)"
    else:
        xlabel = "Frame"

    unique_ids  = df_aligned["COLOCALIZE_ID"].unique()
    id_to_pos   = {cid: idx for idx, cid in enumerate(unique_ids)}

    fig, ax = plt.subplots(figsize=config["tl_figsize"], dpi=config["dpi"])

    C1_data = df_aligned[df_aligned["CHANNEL"] == 1]
    C2_data = df_aligned[df_aligned["CHANNEL"] == 2]

    ax.barh(
        y=C1_data["POSITION"] * 2 + 1,
        width=C1_data["TRACK_DURATION"],
        left=C1_data["ADJUSTED_START"],
        color=config["tl_colorC1"], label=config["C1_name"],
    )
    ax.barh(
        y=C2_data["POSITION"] * 2,
        width=C2_data["TRACK_DURATION"],
        left=C2_data["ADJUSTED_START"],
        color=config["tl_colorC2"], label=config.get("tl_labelC2", "C2"),
    )

    ax.set_yticks(
        ticks=[id_to_pos[cid] * 2 + 0.5 for cid in unique_ids],
        labels=[f"ID {cid}" for cid in unique_ids],
        fontsize=7,
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Track pair", fontsize=12)
    ax.set_title(config.get("tl_title_individual", ""), fontsize=14)
    if config.get("tl_xlim"):
        ax.set_xlim(config["tl_xlim"])
    if config.get("tl_xticks"):
        ax.xaxis.set_major_locator(MultipleLocator(config["tl_xticks"]))
    ax.legend(loc="best")
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    return fig


def _bootstrap_CI_timeline(data, n_bootstraps=1000):
    """
    95% CI for the median via bootstrapping.
    Returns [[lower_error], [upper_error]] format for ax.errorbar xerr.
    """
    medians = [
        np.median(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstraps)
    ]
    medians = np.array(medians)
    orig    = np.median(data)
    return [[orig - np.percentile(medians, 2.5)], [np.percentile(medians, 97.5) - orig]]


def _compute_timeline_summary(df_combined):
    """Returns a tidy summary DataFrame of start/stop statistics per protein."""
    rows = []
    for protein in df_combined["protein_name"].unique():
        g = df_combined[df_combined["protein_name"] == protein]
        starts = g["ADJUSTED_START"].dropna().values
        stops  = g["ADJUSTED_STOP"].dropna().values
        ci_s   = _bootstrap_CI_timeline(starts)
        ci_e   = _bootstrap_CI_timeline(stops)
        med_s  = np.median(starts)
        med_e  = np.median(stops)
        rows.append({
            "Protein":         protein,
            "N tracks":        len(g),
            "Mean start":      round(np.mean(starts),    3),
            "Median start":    round(med_s,              3),
            "Std start":       round(np.std(starts),     3),
            "CI start lower":  round(med_s - ci_s[0][0], 3),
            "CI start upper":  round(med_s + ci_s[1][0], 3),
            "Mean stop":       round(np.mean(stops),     3),
            "Median stop":     round(med_e,              3),
            "Std stop":        round(np.std(stops),      3),
            "CI stop lower":   round(med_e - ci_e[0][0], 3),
            "CI stop upper":   round(med_e + ci_e[1][0], 3),
        })
    return pd.DataFrame(rows)


def plot_average_timelines(loaded_experiments, config):
    """
    Plots average timelines across multiple experiments.

    Two modes controlled by config["tl_mode"]:
      "duration"   — one bar per protein, from median/mean start to median/mean stop,
                     with optional error markers at each end.
      "start_stop" — two boxplots per protein (one for start, one for stop),
                     optionally separated vertically.

    config["tl_split_C1"] controls whether C1 tracks are shown as one aggregated
    row or as separate per-experiment rows labelled "{C1_name} ({C2_name})".

    Parameters:
    - loaded_experiments: list of ExperimentData objects
    - config: dict

    Returns:
    - fig: matplotlib Figure
    - summary_df: pd.DataFrame with statistics (show optionally in Streamlit)
    """
    protein_names_C2 = [exp.protein_name for exp in loaded_experiments]
    split_C1         = config.get("tl_split_C1", False)

    # ── Align and merge ───────────────────────────────────────────────────
    aligned_dfs = []
    for exp in loaded_experiments:
        df_a = align_bars(exp.df_colocalization, zero_at=config["tl_zero_at"]).copy()
        if split_C1:
            # Each experiment's C1 tracks get a unique label
            c1_label = f"{config['C1_name']}-{exp.protein_name}"
        else:
            c1_label = config["C1_name"]
        df_a["protein_name"] = np.where(df_a["CHANNEL"] == 1, c1_label, exp.protein_name)
        aligned_dfs.append(df_a)

    df_combined = pd.concat(aligned_dfs, ignore_index=True)

    # Framerate conversion
    if config["duration_in_seconds"] and config.get("framerate"):
        fr = config["framerate"]
        df_combined[["ADJUSTED_START", "ADJUSTED_STOP", "TRACK_DURATION"]] *= fr
        xlabel = "Time (s)"
    else:
        xlabel = "Frame"

    # ── Build y-axis order and color map ─────────────────────────────────
    # Layout (bottom → top):
    #   Aggregated:  C2 proteins reversed, then single C1 row
    #   Split C1:    C2 proteins reversed, then per-experiment C1 rows
    #                (in same order as experiments, so C1(exp1) is just above C2s)
    c1_color  = config["colors"][0]
    c2_colors = config["colors"][1:]  # one per C2 experiment

    if split_C1:
        c1_labels  = [f"{config['C1_name']}-{name}" for name in protein_names_C2]
        c1_shades  = _red_shades(len(c1_labels))
        # C1 groups first (experiment order), then C2 proteins reversed
        y_labels  = protein_names_C2 + c1_labels
        color_map  = {name: color for name, color in zip(protein_names_C2, c2_colors)}
        color_map.update({label: shade for label, shade in zip(c1_labels, c1_shades)})
    else:
        y_labels  = protein_names_C2 + [config["C1_name"]]
        color_map = {name: color for name, color in zip(protein_names_C2, c2_colors)}
        color_map[config["C1_name"]] = c1_color

    # Summary stats (used by both modes and returned for optional display)
    summary_stats = df_combined.groupby("protein_name", sort=False).agg(
        mean_start    =("ADJUSTED_START",  "mean"),
        mean_stop     =("ADJUSTED_STOP",   "mean"),
        median_start  =("ADJUSTED_START",  "median"),
        median_stop   =("ADJUSTED_STOP",   "median"),
        start_std     =("ADJUSTED_START",  "std"),
        stop_std      =("ADJUSTED_STOP",   "std"),
        n_tracks      =("TRACK_DURATION",  "size"),
    ).reset_index()
    # Reindex to match y_labels order
    summary_stats = summary_stats.set_index("protein_name").reindex(y_labels).reset_index()

    fig, ax = plt.subplots(figsize=config["tl_figsize"], dpi=config["dpi"])

    # ── MODE: duration ────────────────────────────────────────────────────
    if config["tl_mode"] == "duration":

        for idx, row in summary_stats.iterrows():
            protein = row["protein_name"]
            color   = color_map.get(protein, "gray")

            if config["tl_type_central"] == "median":
                bar_start, bar_end = row["median_start"], row["median_stop"]
            else:
                bar_start, bar_end = row["mean_start"], row["mean_stop"]

            bar_length = bar_end - bar_start
            ax.barh(y=idx, width=bar_length, left=bar_start, color=color, alpha=0.7)

            # N annotation
            ax.text(
                1.01, idx, f"N = {int(row['n_tracks'])}",
                va="center", ha="left", fontsize=10, weight="semibold",
                transform=ax.get_yaxis_transform(),
            )

            # Error bars
            if config["tl_plot_error"]:
                prot_data = df_combined[df_combined["protein_name"] == protein]
                starts    = prot_data["ADJUSTED_START"].dropna().values
                stops     = prot_data["ADJUSTED_STOP"].dropna().values

                if config["tl_type_error"] == "ci":
                    err_start = _bootstrap_CI_timeline(starts)
                    err_stop  = _bootstrap_CI_timeline(stops)
                else:
                    err_start = row["start_std"]
                    err_stop  = row["stop_std"]

                ax.errorbar(bar_start, idx, xerr=err_start,
                            fmt="o", color="black", capsize=5, alpha=0.8)
                ax.errorbar(bar_end,   idx, xerr=err_stop,
                            fmt="o", color="black", capsize=5, alpha=0.8)

        # Scatter individual points
        if config["tl_plot_scatter"]:
            _scatter_timeline_points(df_combined, y_labels, config["tl_jitter"], ax)

    # ── MODE: start_stop (boxplots) ───────────────────────────────────────
    else:
        for idx, row in summary_stats.iterrows():
            protein   = row["protein_name"]
            color     = color_map.get(protein, "gray")
            prot_data = df_combined[df_combined["protein_name"] == protein]

            if config["tl_separate_start_stop"]:
                y_start, y_stop = idx + 0.2, idx - 0.2
            else:
                y_start = y_stop = idx

            box_kwargs = dict(
                bootstrap=10000, widths=0.3, vert=False, patch_artist=True,
                boxprops=dict(facecolor=color, color="black", alpha=0.6),
                medianprops=dict(color="black"), showfliers=False,
            )
            ax.boxplot(prot_data["ADJUSTED_START"].dropna(), positions=[y_start], **box_kwargs)
            ax.boxplot(prot_data["ADJUSTED_STOP"].dropna(),  positions=[y_stop],  **box_kwargs)

            ax.text(
                1.01, idx, f"N = {int(row['n_tracks'])}",
                va="center", ha="left", fontsize=10, weight="semibold",
                transform=ax.get_yaxis_transform(),
            )

            if config["tl_plot_scatter"]:
                jitter = config["tl_jitter"]
                s_jit  = y_start + np.random.uniform(-jitter, jitter, len(prot_data))
                e_jit  = y_stop  + np.random.uniform(-jitter, jitter, len(prot_data))
                ax.scatter(prot_data["ADJUSTED_START"], s_jit,
                           color=color, alpha=0.5, edgecolor="k", s=18, marker="^")
                ax.scatter(prot_data["ADJUSTED_STOP"],  e_jit,
                           color=color, alpha=0.5, edgecolor="k", s=18, marker="o")

        if config["tl_plot_scatter"]:
            legend_elements = [
                Line2D([0], [0], marker="^", color="black", linestyle="none",
                       markerfacecolor="none", markersize=6, label="Track start"),
                Line2D([0], [0], marker="o", color="black", linestyle="none",
                       markerfacecolor="none", markersize=6, label="Track stop"),
            ]
            ax.legend(handles=legend_elements, loc="best")

    # ── Shared axes formatting ────────────────────────────────────────────
    if config.get("tl_show_line0"):
        ax.axvline(x=0, color="gray", linestyle="--")

    ax.set_yticks(range(len(y_labels)), y_labels, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(config.get("tl_title_average", ""), fontsize=14)
    if config.get("tl_xlim"):
        ax.set_xlim(config["tl_xlim"])
    if config.get("tl_xticks"):
        ax.xaxis.set_major_locator(MultipleLocator(config["tl_xticks"]))

    sns.despine(left=False, bottom=False)
    plt.tight_layout()

    summary_df = _compute_timeline_summary(df_combined)
    return fig, summary_df


def _scatter_timeline_points(df_combined, y_labels, jitter_strength, ax):
    """Helper: scatter individual start/stop points with jitter for duration mode."""
    for i, row in df_combined.iterrows():
        if row["protein_name"] not in y_labels:
            continue
        y_idx      = y_labels.index(row["protein_name"])
        y_jittered = y_idx + np.random.normal(0, jitter_strength)
        ax.scatter(row["ADJUSTED_START"], y_jittered,
                   color="black", alpha=0.3, s=18, marker="^",
                   label="Track start" if i == 0 else "")
        ax.scatter(row["ADJUSTED_STOP"],  y_jittered,
                   color="black", alpha=0.3, s=18, marker="s",
                   label="Track stop"  if i == 0 else "")
    ax.legend(loc="best")
