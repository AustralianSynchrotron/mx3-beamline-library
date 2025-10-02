from os import path
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bluesky.plan_stubs import mv
from scipy.optimize import curve_fit

from mx3_beamline_library.config import redis_connection
from mx3_beamline_library.devices.motors import md3
from mx3_beamline_library.plans.calibration.trays.image_analysis import ImageAnalysis
from mx3_beamline_library.plans.calibration.trays.plane_frame import PlaneFrame
from mx3_beamline_library.plans.calibration.trays.plate_configs import plate_configs
from mx3_beamline_library.plans.image_analysis import get_image_from_md3_camera
from mx3_beamline_library.plans.plan_stubs import md3_move


def take_md3_image():
    image = get_image_from_md3_camera(np.uint16)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


def autofocus_scan(start, stop, coarse_step, fine_range, fine_step, fit_top_n=5):
    yield from mv(md3.sample_y, start)

    def run_scan(positions):
        dat = []
        for pos in positions:
            yield from mv(md3.sample_y, pos)
            image = get_image_from_md3_camera(np.uint16)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            var = lap.var()
            dat.append({"position": pos, "sharpness": var})
        return pd.DataFrame(dat)

    # --- Coarse Scan ---
    coarse_positions = np.arange(start, stop, coarse_step)
    print("Running coarse scan...")
    df_coarse = yield from run_scan(coarse_positions)
    peak_coarse = df_coarse.loc[df_coarse["sharpness"].idxmax(), "position"]

    # --- Fine Scan ---
    fine_start = (peak_coarse - 0.1) - fine_range / 2
    fine_stop = (peak_coarse - 0.1) + fine_range / 2
    fine_positions = np.arange(fine_start, fine_stop + fine_step / 2, fine_step)
    print("Running fine scan...")
    yield from mv(md3.sample_y, peak_coarse)
    df_fine = yield from run_scan(fine_positions)

    # Fit to top N fine points using quadratic
    best_pos = None
    try:
        df_fine_top = df_fine.sort_values(by="sharpness", ascending=False).head(
            fit_top_n
        )
        x_fine = df_fine_top["position"].values
        y_fine = df_fine_top["sharpness"].values
        coeffs_fine = np.polyfit(x_fine, y_fine, 2)

        # Check for sensible curve
        if coeffs_fine[0] < 0 and abs(coeffs_fine[0]) > 1e-4:
            peak_fine = -coeffs_fine[1] / (2 * coeffs_fine[0])
            print(f"Quadratic fine peak: {peak_fine:.3f}")
            best_pos = peak_fine
            fit_label = "Quadratic"
        else:
            raise ValueError("Quadratic fit too flat or opening up")

    except Exception as e:
        print(f"Quadratic fit failed or rejected: {e}")
        # Fallback to Gaussian fit
        try:
            x_all = df_fine["position"].values
            y_all = df_fine["sharpness"].values
            p0 = [np.max(y_all), x_all[np.argmax(y_all)], 0.1, np.min(y_all)]
            popt, _ = curve_fit(gaussian, x_all, y_all, p0=p0)
            best_pos = popt[1]
            print(f"Gaussian fine peak: {best_pos:.3f}")
            fit_label = "Gaussian"
        except Exception as e2:
            print(f"Gaussian fit also failed: {e2}")
            best_pos = df_fine.loc[df_fine["sharpness"].idxmax(), "position"]
            fit_label = "Raw max"

    # Move to best position
    yield from mv(md3.sample_y, best_pos)
    print(f"Moved to best focus ({fit_label}): {best_pos:.3f}")

    # --- Plot ---
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_coarse["position"],
            y=df_coarse["sharpness"],
            mode="markers+lines",
            name="Coarse Scan",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_fine["position"],
            y=df_fine["sharpness"],
            mode="markers+lines",
            name="Fine Scan",
        )
    )
    fig.update_yaxes(insiderange=[0, 50])

    # Overlay fit if successful
    try:
        if fit_label == "Quadratic":
            x_curve = np.linspace(
                df_fine["position"].min(), df_fine["position"].max(), 200
            )
            y_curve = np.poly1d(coeffs_fine)(x_curve)
            fig.add_trace(
                go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Quadratic Fit")
            )
        elif fit_label == "Gaussian":
            x_curve = np.linspace(
                df_fine["position"].min(), df_fine["position"].max(), 200
            )
            y_curve = gaussian(x_curve, *popt)
            fig.add_trace(
                go.Scatter(x=x_curve, y=y_curve, mode="lines", name="Gaussian Fit")
            )
    except:
        pass

    fig.add_vline(
        x=best_pos,
        line_dash="dash",
        line_color="green",
        annotation_text=f"{fit_label} best: {best_pos:.3f}",
    )
    fig.update_layout(
        title="Autofocus Coarse + Fine Scan",
        xaxis_title="Position",
        yaxis_title="Sharpness",
    )
    # fig.show()


def get_current_position():
    return np.array(
        [
            md3.alignment_y.position,
            md3.plate_translation.position,
            md3.sample_y.position,
        ]
    )


def define_plane_frame(reference_points):
    if "H12" in reference_points:
        A1, H1, H12 = (
            reference_points["A1"],
            reference_points["H1"],
            reference_points["H12"],
        )
        # Define axes
        u_axis = H12 - H1  # columns, x axis, alignment y, vertical in instrument space
        v_axis = (
            H1 - A1
        )  # rows, y axis, plate translation, horizontal in instrument space
        origin = A1

    else:
        A1, H1, A12 = (
            reference_points["A1"],
            reference_points["H1"],
            reference_points["A12"],
        )
        # Define axes
        u_axis = A12 - A1  # columns, x axis, alignment y, vertical in instrument space
        v_axis = (
            H1 - A1
        )  # rows, y axis, plate translation, horizontal in instrument space
        origin = A1
    # Normalize axes
    u_axis /= np.linalg.norm(u_axis)
    v_axis /= np.linalg.norm(v_axis)
    return PlaneFrame(origin=origin, u_axis=u_axis, v_axis=v_axis)


def get_positions(well_label: str, plane: PlaneFrame, config: dict):
    origin_uv = config["origin_uv"]
    dx = config["dx"]
    dy = config["dy"]
    wellx = config["wellx"]
    welly = config["welly"]
    subgrid_rows = config["subgrid_rows"]
    subgrid_cols = config["subgrid_cols"]

    row_char = well_label[0].upper()
    col_num = int(well_label[1:])

    row = ord(row_char) - ord("A")
    col = col_num - 1

    base_u = origin_uv[0] + (col * dx)
    base_v = origin_uv[1] + (row * dy)

    positions = []
    for sr in range(subgrid_rows):
        for sc in range(subgrid_cols):
            u = base_u + sc * wellx
            v = base_v + sr * welly
            motor_pos = plane.local_to_motor(u, v)
            spot_number = sr * subgrid_cols + sc + 1
            positions.append(
                {
                    "motor_pos": tuple(motor_pos),
                    "well": well_label.upper(),
                    "spot": spot_number,
                }
            )
    return positions


def move_to_calibration_spot(well_input: str, plane: PlaneFrame, config: dict):
    import re

    match = re.match(r"([A-Ia-i][0-9]+)", well_input)
    if not match:
        raise ValueError(f"Invalid input: {well_input}. Format should be like 'A1'")

    well_label = match.group(1).upper()
    position = np.array(config["calibration_points"][well_label])

    yield from md3_move(
        md3.alignment_y,
        position[0],
        md3.plate_translation,
        position[1],
        md3.sample_y,
        position[2],
    )

    print(
        f"Moved to {well_label} calibration point: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f}"
    )


def move_to_well_spot(
    well_input: str,
    plate_type: Literal[
        "swissci_lowprofile", "mitegen_insitu", "swissci_highprofile", "mrc"
    ],
):
    """
    Move motors to a specific well + spot, e.g. 'A4 well 4'
    """
    if plate_type == "mitegen_insitu":
        config = plate_configs.mitegen_insitu
    elif plate_type == "swissci_highprofile":
        config = plate_configs.swissci_highprofile
    elif plate_type == "mrc":
        config = plate_configs.mrc
    elif plate_type == "swissci_lowprofile":
        config = plate_configs.swissci_lowprofile
    else:
        raise ValueError(f"Unknown plate type: {plate_type}")
    import re

    match = re.match(r"([A-Ia-i][0-9]+)[\s_]*well[\s_]*(\d)", well_input)
    if not match:
        raise ValueError(
            f"Invalid input: {well_input}. Format should be like 'A4 well 2'"
        )

    well_label = match.group(1).upper()
    spot_num = int(match.group(2))

    if not (1 <= spot_num <= 4):
        raise ValueError("Spot number must be 1 to 4")

    # Get all 4 sub-positions from the redis calibration plane
    res = redis_connection.hgetall("tray_calibration_params")
    if not res:
        raise RuntimeError(
            "No calibration parameters found in Redis. Run calibration first."
        )
    origin = np.array(list(map(float, res[b"origin"].decode().split(","))))
    u_axis = np.array(list(map(float, res[b"u_axis"].decode().split(","))))
    v_axis = np.array(list(map(float, res[b"v_axis"].decode().split(","))))

    plane = PlaneFrame(origin, u_axis, v_axis)

    positions = get_positions(well_label, plane, config)
    selected = positions[spot_num - 1]["motor_pos"]

    # Unpack and move motors
    depth_offset = config["depth"]
    print(depth_offset)
    x, y, z = selected
    yield from md3_move(
        md3.alignment_y, x, md3.plate_translation, y, md3.sample_y, z + depth_offset
    )

    print(f"Moved to {well_label} spot {spot_num}: x={x:.3f}, y={y:.3f}, z={z:.3f}")


def update_reference_points(points, plane, config: dict):
    """
    For each reference point (e.g., A1, H1, A12), move to it,
    refocus, run image analysis, and update its motor position.

    Returns:
        A dictionary mapping point labels to updated motor positions.
    """
    updated_refs = {}
    plate_name = config["type"]

    for point_label in points:
        # Move to the default expected position
        yield from move_to_calibration_spot(f"{point_label} well 1", plane, config)

        # Perform a rough focus
        yield from autofocus_scan(
            start=config["scan"]["start"],
            stop=config["scan"]["stop"],  # Coarse scan range
            coarse_step=0.05,  # Coarse step size
            fine_range=0.1,  # Fine scan window width
            fine_step=0.02,  # Fine scan step
            fit_top_n=5,  # Points used for fitting
        )
        # Capture and analyze image
        reference_filename = path.join(
            path.dirname(__file__),
            f"plate_configs/{plate_name}/plate_{point_label}.png",
        )
        # reference_filename = f"plate_configs/{plate_name}/plate_{point_label}.png"
        analyser = ImageAnalysis()
        analyser.run(reference_filename)

        # Read back actual current motor positions after image alignment
        current_pos = get_current_position()  # Should return np.array([x, y, z])
        updated_refs[point_label] = current_pos
        print(f"{point_label} updated to: {current_pos}")

    return updated_refs


def calibrate_plate(
    plate_type: Literal[
        "swissci_lowprofile", "mitegen_insitu", "swissci_highprofile", "mrc"
    ],
):
    if plate_type == "mitegen_insitu":
        config = plate_configs.mitegen_insitu
    elif plate_type == "swissci_highprofile":
        config = plate_configs.swissci_highprofile
    elif plate_type == "mrc":
        config = plate_configs.mrc
    elif plate_type == "swissci_lowprofile":
        config = plate_configs.swissci_lowprofile
    else:
        raise ValueError(f"Unknown plate type: {plate_type}")
    reference_points = config["reference_points"]
    plane = define_plane_frame(reference_points)
    calibrated_points = yield from update_reference_points(
        reference_points, plane, config
    )
    cal_plane = define_plane_frame(calibrated_points)
    print("Calibration complete")

    redis_connection.hset(
        f"tray_calibration_params",
        mapping={
            "origin": ",".join(map(str, cal_plane.origin)),
            "u_axis": ",".join(map(str, cal_plane.u_axis)),
            "v_axis": ",".join(map(str, cal_plane.v_axis)),
        },
    )

    return cal_plane
