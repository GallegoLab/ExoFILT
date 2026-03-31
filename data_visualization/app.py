import io as _io
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for Streamlit

from data_loader import load_all_experiments
from pipeline import (
    process_experiments,
    data_individual_profiles,
    plot_average_profile,
    plot_individual_profiles,
    plot_individual_timelines,
    plot_average_timelines,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be the first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dual-color Tracking Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    section[data-testid="stSidebar"] { padding-top: 1rem; }
    .experiment-card {
        background: #f8f9fa;
        border-left: 3px solid #aaa;
        border-radius: 4px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.88rem;
    }
    .experiment-card b { font-size: 0.95rem; }
    div[data-testid="stAlert"] { padding: 0.4rem 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

defaults = {
    "experiments":           [],
    "loaded_experiments":    [],
    "experiment_visibility": {},   # {protein_name: bool} — True = included in analysis
    "results_ready":         False,
    "averaged_data":         None,
    "individual_data_list":  [],
    "load_messages":         [],
    "last_config":           None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_COLORS = [
    "#e74c3c",  # red      — C1 reference
    "#2980b9",  # steel blue
    "#27ae60",  # green
    "#e67e22",  # orange
    "#8e44ad",  # purple
    "#16a085",  # teal
    "#c0392b",  # dark red
    "#2c3e50",  # dark slate
]

def next_default_color() -> str:
    used = {exp["color"] for exp in st.session_state.experiments}
    for color in DEFAULT_COLORS:
        if color not in used:
            return color
    return DEFAULT_COLORS[len(st.session_state.experiments) % len(DEFAULT_COLORS)]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Intensity Profiles")
    st.divider()

    # ── 1. Add experiment ───────────────────────────────────────────────────

    st.subheader("Add experiment")

    protein_name_input = st.text_input(
        "Protein name (C2 channel)",
        placeholder="e.g. Sec1, Sec9, Myo2 ...",
        key="input_protein_name",
    )
    col_csv_input = st.file_uploader(
        "Colocalization CSV", type=["csv"], key="input_csv",
        help="Colocalization output CSV for this protein.",
    )
    zip_input = st.file_uploader(
        "Intensity profiles (zip)", type=["zip"], key="input_zip",
        help="Zip containing all Colocalized_ID_*_C*.csv files.",
    )
    color_input = st.color_picker(
        "Trace color", value=next_default_color(), key="input_color",
    )

    existing_names = [e["protein_name"] for e in st.session_state.experiments]
    name_clash = protein_name_input.strip() and protein_name_input.strip() in existing_names
    can_add = (
        bool(protein_name_input.strip())
        and col_csv_input is not None
        and zip_input is not None
        and not name_clash
    )
    if name_clash:
        st.warning(f"'{protein_name_input.strip()}' is already in the list.")

    if st.button("➕ Add experiment", disabled=not can_add, use_container_width=True):
        st.session_state.experiments.append({
            "protein_name":         protein_name_input.strip(),
            "color":                color_input,
            "colocalization_bytes": col_csv_input.read(),
            "intensity_zip_bytes":  zip_input.read(),
        })
        st.session_state.results_ready  = False
        st.session_state.loaded_experiments = []
        st.session_state.averaged_data  = None
        st.rerun()

    # ── 2. Experiment list ──────────────────────────────────────────────────

    st.divider()
    st.subheader("Experiments")

    if not st.session_state.experiments:
        st.caption("No experiments added yet.")
    else:
        for i, exp in enumerate(st.session_state.experiments):
            name = exp["protein_name"]

            # Ensure every experiment has a visibility entry (handles newly added ones)
            if name not in st.session_state.experiment_visibility:
                st.session_state.experiment_visibility[name] = True

            is_visible = st.session_state.experiment_visibility[name]

            # Dim the card when hidden
            card_opacity = "1.0" if is_visible else "0.4"
            swatch_color = exp["color"] if is_visible else "#aaaaaa"

            col_toggle, col_info, col_btn = st.columns([1, 5, 1])

            with col_toggle:
                # Eye toggle — clicking flips visibility and marks analysis stale
                toggled = st.checkbox(
                    "##",
                    value=is_visible,
                    key=f"vis_{name}_{i}",
                    help="Include in analysis" if is_visible else "Excluded from analysis",
                    label_visibility="collapsed",
                )
                if toggled != is_visible:
                    st.session_state.experiment_visibility[name] = toggled
                    st.session_state.averaged_data = None   # results are now stale
                    st.rerun()

            with col_info:
                swatch = (
                    f'<span style="display:inline-block;width:10px;height:10px;'
                    f'border-radius:50%;background:{swatch_color};'
                    f'margin-right:6px;vertical-align:middle;"></span>'
                )
                label = name if is_visible else f"<i>{name} (hidden)</i>"
                st.markdown(
                    f'<div class="experiment-card" style="opacity:{card_opacity};">'
                    f'{swatch}<b>{label}</b></div>',
                    unsafe_allow_html=True,
                )

            with col_btn:
                if st.button("✕", key=f"remove_{i}", help=f"Remove {name}"):
                    st.session_state.experiments.pop(i)
                    st.session_state.experiment_visibility.pop(name, None)
                    st.session_state.results_ready = False
                    st.session_state.loaded_experiments = []
                    st.session_state.averaged_data = None
                    st.rerun()
                    
    c1_name = st.text_input("C1 protein name", value="Exo84")

    # ── 3. Load & validate ──────────────────────────────────────────────────

    st.divider()

    if st.button(
        "▶ Load & validate data",
        disabled=len(st.session_state.experiments) == 0,
        use_container_width=True,
        type="primary",
    ):
        with st.spinner("Loading experiments..."):
            loaded, messages = load_all_experiments(st.session_state.experiments)
        st.session_state.loaded_experiments = loaded
        st.session_state.load_messages      = messages
        st.session_state.results_ready      = len(loaded) > 0
        st.session_state.averaged_data      = None
        st.rerun()

    # ── 4. Timeline configuration (always live — no Run button needed) ──────

    if st.session_state.results_ready:
        st.divider()
        st.subheader("Configuration: timeline")

        st.markdown("**Alignment & time**")
        _, col = st.columns([0.08, 0.92])
        with col:
            tl_zero_at = st.selectbox(
                "Align zero at",
                options=["C1_start", "C1_stop", "C1_center",
                         "C2_start", "C2_stop", "C2_center"],
                key="tl_zero_at",
                help="Independent from the intensity profile alignment.",
            )

        st.markdown("**Individual timelines**")
        _, col = st.columns([0.08, 0.92])
        with col:
            tl_sort_channel   = st.selectbox("Sort by channel", [1, 2], key="tl_sort_ch")
            tl_sort_ascending = st.checkbox("Sort ascending (short→long)", value=False,
                                            key="tl_sort_asc")

        with st.expander("**Average timelines**"):
            tl_split_C1 = st.checkbox(
                "Split C1 by experiment", value=False, key="tl_split_c1",
                help=f"Show one C1 row per experiment (e.g. Exo84-Sec1, Exo84-Myo2) "
                     f"instead of pooling all C1 tracks together.",
            )
            tl_mode = st.radio(
                "Plot type",
                options=["duration", "start_stop"],
                format_func=lambda x: "Duration bar" if x == "duration" else "Start / Stop boxplots",
                horizontal=True, key="tl_mode",
            )
            tl_type_central = st.selectbox(
                "Central tendency", ["median", "mean"],
                key="tl_central",
                disabled=(tl_mode != "duration"),
            )
            tl_plot_error = st.checkbox(
                "Show error bars", value=True, key="tl_plot_error",
                disabled=(tl_mode != "duration"),
            )
            tl_type_error = st.selectbox(
                "Error type", ["ci", "std"], key="tl_error_type",
                disabled=(tl_mode != "duration" or not tl_plot_error),
                help="ci = 95% bootstrap CI · std = standard deviation",
            )
            tl_separate_start_stop = st.checkbox(
                "Separate start/stop rows", value=True, key="tl_separate",
                disabled=(tl_mode != "start_stop"),
            )

        with st.expander("**Scatter & overlays**"):
            tl_plot_scatter    = st.checkbox("Show individual points", value=True, key="tl_scatter")
            tl_jitter          = st.slider("Jitter strength", 0.0, 0.3, 0.05, 0.01,
                                           key="tl_jitter", disabled=not tl_plot_scatter)
            tl_show_line0      = st.checkbox("Show line at x = 0", value=True, key="tl_line0")
            tl_display_summary = st.checkbox("Show summary statistics table", value=False,
                                             key="tl_summary")

        with st.expander("**Axes & figure**"):
            tl_xlim_auto = st.checkbox("Automatic X axis", value=True, key="tl_xlim_auto")
            if not tl_xlim_auto:
            	col_tl_xl, col_tl_xr = st.columns(2)
            	tl_xlim_left  = col_tl_xl.number_input("X min", value=-10.0, step=1.0, key="tl_xmin")
            	tl_xlim_right = col_tl_xr.number_input("X max", value=20.0,  step=1.0, key="tl_xmax")
            
            tl_xticks = st.number_input("X tick interval (0 = auto)", value=0.0,
                                         step=0.5, min_value=0.0, key="tl_xticks")
            
            col_tl_w, col_tl_h = st.columns(2)
            tl_fig_w = col_tl_w.number_input("Width (in)",  value=8, min_value=2,
                                              max_value=20, key="tl_fw")
            tl_fig_h = col_tl_h.number_input("Height (in)", value=5, min_value=2,
                                              max_value=20, key="tl_fh")
            
            tl_title_individual = st.text_input("Individual timelines title", value="",
                                                key="tl_title_ind")
            tl_title_average    = st.text_input("Average timelines title", value="",
                                                key="tl_title_avg")


    # ── 5. Intensity profile configuration (gated behind Run button) ───────

    if st.session_state.results_ready:
        st.divider()
        st.subheader("Configuration: Intensity Profiles")
        
        split_C1 = st.checkbox(
            "Split C1 by experiment",
            value=False,
            help="Shows a separate C1 average trace per experiment using distinct "
                 "red shades. When off, all C1 tracks are pooled into one trace.",
        )
        
        # Time axis
        st.markdown("**Time axis**")
        _, col = st.columns([0.08, 0.92])
        with col:
            zero_at = st.selectbox(
                "Align zero at",
                options=["C1_start", "C1_stop", "C1_center",
                         "C2_start", "C2_stop", "C2_center"],
            )
            duration_in_seconds = st.checkbox("Show time in seconds", value=True)
            framerate = st.number_input(
                "Framerate (s / frame)",
                min_value=0.001, max_value=10.0,
                value=0.120, step=0.001, format="%.3f",
                disabled=not duration_in_seconds,
            )

        # Normalization
        st.markdown("**Normalization**")
        _, col = st.columns([0.08, 0.92])
        with col:
            normalize_intensity = st.checkbox("Normalize intensity", value=False)
            normalize_duration  = st.checkbox("Normalize duration",  value=False)
            normalize_by_C2     = st.checkbox(
                "Normalize by C2 track length", value=False,
                disabled=not normalize_duration,
                help="Use C2 track duration as the normalization reference.",
            )

        # Smoothing
        st.markdown("**Smoothing**")
        _, col = st.columns([0.08, 0.92])
        with col:
            smooth_intensity = st.checkbox("Smooth intensity", value=True)
            smooth_sigma = st.slider(
                "Gaussian sigma", min_value=0.5, max_value=10.0,
                value=2.0, step=0.5, disabled=not smooth_intensity,
            )

        # Average profile — axes
        with st.expander("**Average profile — axes**"):
            col_xl, col_xr = st.columns(2)
            xlim_left  = col_xl.number_input("X min", value=-5.0, step=1.0)
            xlim_right = col_xr.number_input("X max", value=15.0,  step=1.0)
            col_yl, col_yr = st.columns(2)
            ylim_bottom = col_yl.number_input("Y min", value=-5.0,  step=1.0)
            ylim_top    = col_yr.number_input("Y max", value=50.0,  step=1.0)
            xticks = st.number_input(
                "X tick interval (0 = auto)", value=0.0, step=0.5, min_value=0.0,
            )

        # Average profile — overlays
        with st.expander("**Average profile — overlays**"):
            plot_intensity_error = st.selectbox(
                "Show intensity error", ["None", "Standard deviation", "Confidence interval (95%)"],
            )
            plot_start_stop_central = st.checkbox("Show mean/median start–stop", value=False)
            type_start_stop_central = st.selectbox(
                "Central tendency", ["median", "mean"],
                disabled=not plot_start_stop_central,
            )
            alpha_central = st.slider(
                "Start–stop line opacity", 0.0, 1.0, 0.5, 0.05,
                disabled=not plot_start_stop_central,
            )
            plot_start_stop_error = st.checkbox(
                "Show start–stop error bars", value=True,
                disabled=not plot_start_stop_central,
            )
            type_start_stop_error = st.selectbox(
                "Error type", ["ci", "std"],
                disabled=not (plot_start_stop_central and plot_start_stop_error),
                help="ci = 95% bootstrap CI for the median · std = standard deviation",
            )
            alpha_error = st.slider(
                "Error bar opacity", 0.0, 1.0, 0.5, 0.05,
                disabled=not (plot_start_stop_central and plot_start_stop_error),
            )

        with st.expander("**Individual profiles configuration**"):
            extra_frames       = st.slider("Extra frames around track", 0, 100, 20)
            columns_individual = st.slider("Subplots per row", 1, 8, 3)
            plot_titles     = st.checkbox("Show subplot titles",       value=True)
            plot_duration   = st.checkbox("Show duration on subplots", value=False)
            plot_thresholds = st.checkbox("Show start/stop lines",     value=True)

        with st.expander("**Figure size, resolution, and titles**"):
            dpi = st.select_slider("DPI", options=[72, 100, 150, 200, 300], value=150)

            st.markdown("**Average profile**")
            col_fw_avg, col_fh_avg = st.columns(2)
            fig_w_avg = col_fw_avg.number_input("Width (in)",  value=5, min_value=1, max_value=20, key="fig_w_avg")
            fig_h_avg = col_fh_avg.number_input("Height (in)", value=4, min_value=1, max_value=20, key="fig_h_avg")
            title_average = st.text_input("Title", value="", key="title_avg")

            st.markdown("**Individual profiles**")
            col_fw_ind, col_fh_ind = st.columns(2)
            fig_w_ind = col_fw_ind.number_input("Width (in)",  value=4, min_value=1, max_value=20, key="fig_w_ind")
            fig_h_ind = col_fh_ind.number_input("Height (in)", value=3, min_value=1, max_value=16, key="fig_h_ind")
            title_figure = st.text_input("Title", value="", key="title_ind")

        st.divider()

        # ── Run analysis ────────────────────────────────────────────────────

        if st.button("▶▶ Run analysis (intensity profiles)", use_container_width=True, type="primary"):

            # Build visible list first — needed for colors in config
            visible_experiments = [
                exp for exp in st.session_state.loaded_experiments
                if st.session_state.experiment_visibility.get(exp.protein_name, True)
            ]

            if not visible_experiments:
                st.error("All experiments are hidden — enable at least one to run the analysis.")
                st.stop()

            c2_colors = [exp.color for exp in visible_experiments]

            config = {
                # Identity
                "C1_name": c1_name,
                "split_C1_by_experiment": split_C1,
                # Time
                "zero_at":             zero_at,
                "duration_in_seconds": duration_in_seconds,
                "framerate":           framerate,
                # Normalization
                "normalize_intensity": normalize_intensity,
                "normalize_duration":  normalize_duration,
                "normalize_by_C2":     normalize_by_C2,
                # Smoothing
                "smooth_intensity": smooth_intensity,
                "smooth_sigma":     smooth_sigma,
                # Average plot — axes
                "xlim":   (xlim_left, xlim_right),
                "ylim":   (ylim_bottom, ylim_top),
                "xticks": xticks if xticks > 0 else None,
                # Average plot — overlays
                "plot_intensity_error":     plot_intensity_error,
                "plot_start_stop_central":  plot_start_stop_central,
                "type_start_stop_central":  type_start_stop_central,
                "alpha_start_stop_central": alpha_central,
                "plot_start_stop_error":    plot_start_stop_error,
                "type_start_stop_error":    type_start_stop_error,
                "alpha_start_stop_error":   alpha_error,
                # Individual profiles
                "extra_frames":       extra_frames,
                "columns_individual": columns_individual,
                "figsize_individual": (fig_w_ind, fig_h_ind),
                "plot_titles":        plot_titles,
                "plot_duration":      plot_duration,
                "plot_thresholds":    plot_thresholds,
                # Figure
                "figsize":         (fig_w_avg, fig_h_avg),
                "dpi":             dpi,
                "title_average":   title_average,
                "title_figure":    title_figure,
                "fontsize_legend": 8,
                # Colors: C1 (red) first, then one per C2 experiment
                "colors": [DEFAULT_COLORS[0]] + c2_colors,
                # protein_names is filled in by process_experiments()
                "savefig_path": None,
            }

            with st.spinner("Running analysis..."):
                averaged_data, individual_data_list = process_experiments(
                    loaded_experiments=visible_experiments,
                    config=config,
                )

            st.session_state.averaged_data        = averaged_data
            st.session_state.individual_data_list = individual_data_list
            st.session_state.last_config          = config
            st.rerun()



# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────

st.header("Intensity Profile Viewer")

# ── Gate: nothing added ───────────────────────────────────────────────────────

if not st.session_state.experiments:
    st.info(
        "👈 **Get started** — add at least one experiment in the sidebar.\n\n"
        "Each experiment needs:\n"
        "- A **protein name** for the C2 channel\n"
        "- A **colocalization CSV** file\n"
        "- A **zip file** containing all intensity profile CSVs"
    )
    st.stop()

# ── Gate: not yet loaded ──────────────────────────────────────────────────────

if not st.session_state.results_ready:
    st.info(
        f"{len(st.session_state.experiments)} experiment(s) staged. "
        "Click **▶ Load & validate data** in the sidebar."
    )
    st.stop()

# ── Load messages ─────────────────────────────────────────────────────────────

for msg in st.session_state.load_messages:
    if msg.startswith("❌"):
        st.error(msg)
    elif msg.startswith("⚠️"):
        st.warning(msg)

loaded = st.session_state.loaded_experiments
if not loaded:
    st.error("No experiments loaded successfully. Check the errors above.")
    st.stop()

# ── Data summary ──────────────────────────────────────────────────────────────

st.success(f"✅ {len(loaded)} experiment(s) loaded — configure settings and click **▶▶ Run analysis**.")

col_table, _ = st.columns([2, 1])
with col_table:
    visibility = st.session_state.experiment_visibility
    st.dataframe(
        {
            "Protein (C2)":    [e.protein_name for e in loaded],
            "Included":        ["✅" if visibility.get(e.protein_name, True) else "⬜ hidden"
                                for e in loaded],
            "Tracks":          [len(e.colocalize_ids) for e in loaded],
            "Intensity files": [len(e.intensity_files) for e in loaded],
            "Color":           [e.color for e in loaded],
        },
        hide_index=True,
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
tab_tl_individual, tab_tl_average, tab_avg, tab_individual = st.tabs([
    "⏱ Individual timelines",
    "📊 Average timelines",
    "📈 Average profiles",
    "🔍 Individual profiles",
])

# ── Tabs 1 & 2 — live, no Run button needed ───────────────────────────────────
# tl_config is assembled fresh on every rerun from the live sidebar widgets.
# Timeline tabs are always available once data is loaded.

with tab_tl_individual:
    if not st.session_state.results_ready:
        st.info("👈 Load data first using **▶ Load & validate data**.")
    else:
        loaded_experiments = st.session_state.loaded_experiments

        if len(loaded_experiments) > 1:
            tl_selected_name = st.selectbox(
                "Select experiment",
                [exp.protein_name for exp in loaded_experiments],
                key="tl_ind_selector",
            )
            tl_exp = next(e for e in loaded_experiments if e.protein_name == tl_selected_name)
        else:
            tl_exp = loaded_experiments[0]

        # Build tl_config live from current sidebar widget values
        tl_config = {
            "C1_name":            c1_name,
            "tl_zero_at":         tl_zero_at,
            "tl_sort_channel":    tl_sort_channel,
            "tl_sort_ascending":  tl_sort_ascending,
            "tl_labelC2":         tl_exp.protein_name,
            "tl_colorC1":         DEFAULT_COLORS[0],
            "tl_colorC2":         tl_exp.color,
            "duration_in_seconds": duration_in_seconds,
            "framerate":          framerate,
            "tl_xlim": 		  None if tl_xlim_auto else (tl_xlim_left, tl_xlim_right),
            "tl_xticks": 	  tl_xticks if tl_xticks > 0 else None,
            "tl_figsize":          (tl_fig_w, tl_fig_h),
            "tl_title_individual": tl_title_individual,
            "dpi":                 dpi,
        }

        with st.spinner("Plotting individual timelines..."):
            fig_tl_ind = plot_individual_timelines(tl_exp.df_colocalization, tl_config)

        n_pairs = tl_exp.df_colocalization["COLOCALIZE_ID"].nunique()
        st.caption(f"{n_pairs} track pairs · {tl_exp.protein_name}")
        st.pyplot(fig_tl_ind, use_container_width=False)

        buf_tl_ind = _io.BytesIO()
        fig_tl_ind.savefig(buf_tl_ind, format="pdf", dpi=dpi, bbox_inches="tight")
        buf_tl_ind.seek(0)
        st.download_button(
            label="⬇ Download as PDF",
            data=buf_tl_ind,
            file_name=f"individual_timelines_{tl_exp.protein_name}.pdf",
            mime="application/pdf",
            key="dl_tl_ind",
        )

with tab_tl_average:
    if not st.session_state.results_ready:
        st.info("👈 Load data first using **▶ Load & validate data**.")
    else:
        visible_experiments = [
            exp for exp in st.session_state.loaded_experiments
            if st.session_state.experiment_visibility.get(exp.protein_name, True)
        ]

        if not visible_experiments:
            st.warning("All experiments are hidden. Enable at least one in the sidebar.")
        else:
            c2_colors_tl = [exp.color for exp in visible_experiments]

            # Build tl_config live from current sidebar widget values
            tl_config = {
                "C1_name":               c1_name,
                "tl_zero_at":            tl_zero_at,
                "tl_mode":               tl_mode,
                "tl_type_central":       tl_type_central,
                "tl_plot_error":         tl_plot_error,
                "tl_type_error":         tl_type_error,
                "tl_separate_start_stop": tl_separate_start_stop,
                "tl_plot_scatter":       tl_plot_scatter,
                "tl_jitter":             tl_jitter,
                "tl_show_line0":         tl_show_line0,
                "tl_display_summary":    tl_display_summary,
                "tl_split_C1":           tl_split_C1,
                "duration_in_seconds":   duration_in_seconds,
                "framerate":             framerate,
            	"tl_xlim": 		None if tl_xlim_auto else (tl_xlim_left, tl_xlim_right),
                "tl_xticks": tl_xticks if tl_xticks > 0 else None,
                "tl_figsize":            (tl_fig_w, tl_fig_h),
                "tl_title_average":      tl_title_average,
                "dpi":                   dpi,
                # Colors: C1 first, then one per visible C2 experiment
                "colors": [DEFAULT_COLORS[0]] + c2_colors_tl,
            }

            with st.spinner("Computing average timelines..."):
                fig_tl_avg, summary_df = plot_average_timelines(visible_experiments, tl_config)

            st.pyplot(fig_tl_avg, use_container_width=False)

            buf_tl_avg = _io.BytesIO()
            fig_tl_avg.savefig(buf_tl_avg, format="pdf", dpi=dpi, bbox_inches="tight")
            buf_tl_avg.seek(0)
            st.download_button(
                label="⬇ Download as PDF",
                data=buf_tl_avg,
                file_name="average_timelines.pdf",
                mime="application/pdf",
                key="dl_tl_avg",
            )

            if tl_display_summary:
                st.divider()
                st.subheader("Summary statistics")
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

# ── Tabs 3 & 4 — gated behind Run analysis ───────────────────────────────────
# These use st.session_state.last_config which is only set when Run is clicked.

with tab_avg:
    if st.session_state.averaged_data is None:
        st.info("👈 Configure settings and click **▶▶ Run analysis (intensity profiles)** to see results.")
    else:
        config    = st.session_state.last_config
        fig_avg   = plot_average_profile(st.session_state.averaged_data, config)
        st.pyplot(fig_avg, use_container_width=False)

        buf_avg = _io.BytesIO()
        fig_avg.savefig(buf_avg, format="pdf", dpi=config["dpi"], bbox_inches="tight")
        buf_avg.seek(0)
        st.download_button(
            label="⬇ Download as PDF",
            data=buf_avg,
            file_name="average_profiles.pdf",
            mime="application/pdf",
        )

with tab_individual:
    if st.session_state.averaged_data is None:
        st.info("👈 Configure settings and click **▶▶ Run analysis (intensity profiles)** to see results.")
    else:
        config             = st.session_state.last_config
        loaded_experiments = st.session_state.loaded_experiments
        visibility         = st.session_state.experiment_visibility

        if len(loaded_experiments) > 1:
            selected_name = st.selectbox(
                "Select experiment",
                [exp.protein_name for exp in loaded_experiments],
                format_func=lambda name: (
                    name if visibility.get(name, True)
                    else f"{name}  (hidden from average)"
                ),
            )
            exp = next(e for e in loaded_experiments if e.protein_name == selected_name)
        else:
            exp = loaded_experiments[0]

        if not visibility.get(exp.protein_name, True):
            st.warning(
                f"**{exp.protein_name}** is hidden from the average profile. "
                "Individual tracks are still shown here."
            )

        with st.spinner("Preparing individual profiles..."):
            exp_data = data_individual_profiles(
                df_colocalization=exp.df_colocalization,
                intensity_files=exp.intensity_files,
                config=config,
            )

        n_tracks = len(exp_data["intensities_C1"])
        st.caption(f"{n_tracks} tracks · {exp.protein_name}")

        fig_ind = plot_individual_profiles(exp_data, config)
        st.pyplot(fig_ind, use_container_width=False)

        buf_ind = _io.BytesIO()
        fig_ind.savefig(buf_ind, format="pdf", dpi=config["dpi"], bbox_inches="tight")
        buf_ind.seek(0)
        st.download_button(
            label="⬇ Download as PDF",
            data=buf_ind,
            file_name=f"individual_profiles_{exp.protein_name}.pdf",
            mime="application/pdf",
        )
