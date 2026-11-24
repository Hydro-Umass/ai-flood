from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from pysheds.grid import Grid

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEM_PATH = PROJECT_ROOT / "data" / "houston_dem.tif"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_NODATA = -9999.0
D8_OFFSETS = (
    (64, -1, 0),
    (128, -1, 1),
    (1, 0, 1),
    (2, 1, 1),
    (4, 1, 0),
    (8, 1, -1),
    (16, 0, -1),
    (32, -1, -1),
)


def read_dem_file(filename: Path) -> tuple[np.ma.MaskedArray, dict[str, Any], Any]:
    """Read the first DEM band and return its data, raster profile, and NoData value."""
    if not filename.is_file():
        raise FileNotFoundError(f"DEM file not found: {filename}")
    with rasterio.open(filename) as dataset:
        if dataset.count < 1:
            raise ValueError(f"DEM contains no raster bands: {filename}")
        return (
            dataset.read(1, masked=True),
            dataset.profile.copy(),
            dataset.nodata,
        )


def calculate_flow_data(
    filename: Path, nodata: float | int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Condition a DEM and return conditioned elevations, D8 directions, and accumulation."""
    grid_nodata = nodata if nodata is not None else np.nan
    grid = Grid.from_raster(str(filename), nodata=grid_nodata)
    dem = grid.read_raster(str(filename), nodata=grid_nodata)
    filled_dem = grid.fill_depressions(dem)
    conditioned_dem = grid.resolve_flats(filled_dem)
    flow_direction = grid.flowdir(conditioned_dem)
    accumulation = grid.accumulation(flow_direction, routing="D8")

    return (
        np.asarray(np.ma.filled(conditioned_dem, np.nan), dtype=np.float32),
        np.asarray(np.ma.filled(flow_direction, 0), dtype=np.float32),
        np.asarray(np.ma.filled(accumulation, np.nan), dtype=np.float64),
    )


def calculate_hand(
    elevation: np.ndarray,
    flow_direction: np.ndarray,
    accumulation: np.ndarray,
    valid_dem: np.ndarray,
    stream_threshold: int,
) -> tuple[np.ndarray, int, int]:
    """Calculate elevation above the first downstream stream cell for each DEM cell."""
    if stream_threshold < 1:
        raise ValueError("Stream threshold must be a positive number of cells.")
    if not (
        elevation.ndim == flow_direction.ndim == accumulation.ndim == valid_dem.ndim == 2
        and elevation.shape == flow_direction.shape == accumulation.shape == valid_dem.shape
    ):
        raise ValueError("Elevation, flow, accumulation, and validity arrays must match.")

    rows, columns = elevation.shape
    elevation_flat = elevation.ravel()
    direction_flat = flow_direction.ravel()
    accumulation_flat = accumulation.ravel()
    valid_flat = (
        valid_dem.ravel()
        & np.isfinite(elevation_flat)
        & np.isfinite(accumulation_flat)
        & (accumulation_flat >= 1)
    )
    stream_flat = valid_flat & (accumulation_flat >= stream_threshold)
    stream_count = int(stream_flat.sum())
    if stream_count == 0:
        raise ValueError(
            f"No stream cells meet the accumulation threshold of {stream_threshold}."
        )

    downstream = np.full(elevation_flat.size, -1, dtype=np.int64)
    for code, row_offset, column_offset in D8_OFFSETS:
        source_indices = np.flatnonzero(valid_flat & (direction_flat == code))
        source_rows = source_indices // columns
        source_columns = source_indices % columns
        target_rows = source_rows + row_offset
        target_columns = source_columns + column_offset
        inside_grid = (
            (target_rows >= 0)
            & (target_rows < rows)
            & (target_columns >= 0)
            & (target_columns < columns)
        )
        source_indices = source_indices[inside_grid]
        target_indices = target_rows[inside_grid] * columns + target_columns[inside_grid]
        valid_targets = valid_flat[target_indices]
        source_indices = source_indices[valid_targets]
        target_indices = target_indices[valid_targets]

        if np.any(accumulation_flat[target_indices] <= accumulation_flat[source_indices]):
            raise ValueError(
                "Flow accumulation does not increase downstream; check the D8 "
                "directions and accumulation raster."
            )
        downstream[source_indices] = target_indices

    valid_indices = np.flatnonzero(valid_flat)
    descending_order = valid_indices[
        np.argsort(accumulation_flat[valid_indices], kind="stable")[::-1]
    ]
    channel_elevation = np.full(elevation_flat.size, np.nan, dtype=np.float32)
    channel_elevation[stream_flat] = elevation_flat[stream_flat]

    for cell_index in descending_order:
        if stream_flat[cell_index]:
            continue
        downstream_index = downstream[cell_index]
        if downstream_index >= 0:
            channel_elevation[cell_index] = channel_elevation[downstream_index]

    reaches_stream = valid_flat & np.isfinite(channel_elevation)
    hand = np.full(elevation_flat.size, np.nan, dtype=np.float32)
    hand[reaches_stream] = (
        elevation_flat[reaches_stream] - channel_elevation[reaches_stream]
    )
    return hand.reshape(elevation.shape), stream_count, int(reaches_stream.sum())


def save_hand_raster(
    hand: np.ndarray, profile: dict[str, Any], output_path: Path, threshold: int
) -> None:
    """Write the HAND values with the source DEM's georeferencing."""
    output_profile = profile.copy()
    output_profile.update(
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )
    output_values = np.where(np.isfinite(hand), hand, OUTPUT_NODATA).astype(
        np.float32
    )
    with rasterio.open(output_path, "w", **output_profile) as dataset:
        dataset.write(output_values, 1)
        dataset.set_band_description(1, "Height above nearest downstream drainage")
        dataset.update_tags(
            method="HAND: conditioned DEM elevation minus downstream stream elevation",
            stream_accumulation_threshold_cells=str(threshold),
        )


def read_raster_band(filename: Path, band_number: int) -> np.ma.MaskedArray:
    """Read one raster band from a file, masking its NoData pixels."""
    if not filename.is_file():
        raise FileNotFoundError(f"Raster file not found: {filename}")
    with rasterio.open(filename) as dataset:
        if not 1 <= band_number <= dataset.count:
            raise ValueError(
                f"Band {band_number} is not available in {filename}; "
                f"the raster contains {dataset.count} band(s)."
            )
        return dataset.read(band_number, masked=True)


def plot_hand_raster(
    hand_band_data: np.ma.MaskedArray, output_path: Path, threshold: int
) -> None:
    """Plot HAND values, masking cells that do not drain to the selected stream network."""
    hand_masked = np.ma.masked_invalid(hand_band_data)
    values = hand_masked.compressed()
    if values.size == 0:
        raise ValueError("HAND has no valid cells to plot.")
    minimum, maximum = np.percentile(values, (2, 98))
    if minimum == maximum:
        minimum = float(values.min())
        maximum = float(values.max())

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(
        hand_masked,
        cmap="terrain",
        vmin=minimum,
        vmax=maximum if maximum > minimum else minimum + 1,
    )
    axis.set_title(f"Height Above Nearest Drainage (threshold {threshold:,} cells)")
    axis.set_axis_off()
    figure.colorbar(image, ax=axis, label="Height above drainage (DEM elevation units)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Height Above Nearest Drainage (HAND) from a DEM."
    )
    parser.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM_PATH,
        help=f"Input DEM GeoTIFF (default: {DEFAULT_DEM_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1000,
        help="Minimum D8 flow-accumulation cells that define drainage (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for the HAND raster and preview (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dem_data, profile, nodata = read_dem_file(arguments.dem)
    filled_dem, flow_direction, accumulation = calculate_flow_data(
        arguments.dem, nodata
    )
    valid_dem = ~np.ma.getmaskarray(dem_data) & np.isfinite(filled_dem)
    hand, stream_count, assigned_count = calculate_hand(
        filled_dem,
        flow_direction,
        accumulation,
        valid_dem,
        arguments.threshold,
    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    filename_stem = f"{arguments.dem.stem}_hand_threshold_{arguments.threshold}"
    raster_path = arguments.output_dir / f"{filename_stem}.tif"
    plot_path = arguments.output_dir / f"{filename_stem}.png"
    save_hand_raster(hand, profile, raster_path, arguments.threshold)
    hand_band_data = read_raster_band(raster_path, band_number=1)
    plot_hand_raster(hand_band_data, plot_path, arguments.threshold)

    print(f"Stream cells at threshold: {stream_count:,}")
    print(f"DEM cells assigned a HAND value: {assigned_count:,}")
    print(f"Saved HAND raster: {raster_path}")
    print(f"Saved HAND preview: {plot_path}")


if __name__ == "__main__":
    main()
