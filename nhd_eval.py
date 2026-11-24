from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DEM_PATH = DATA_DIR / "houston_dem.tif"
ACCUMULATION_PATH = RESULTS_DIR / "flow_accumulation.tif"
NHD_PATH = DATA_DIR / "nhd_flowlines_houston.geojson"
MAP_PATH = RESULTS_DIR / "nhd_stream_network_comparison.png"
SUMMARY_PATH = RESULTS_DIR / "nhd_comparison_summary.txt"

NHD_LAYER_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6/query"
)
NHD_LAYER_DESCRIPTION = (
    "USGS National Hydrography Dataset (NHD), Flowline - Large Scale, layer 6"
)
STREAM_THRESHOLDS = (1000, 5000, 25000)
DISTANCE_TOLERANCES_M = (25, 50, 100, 250)
COMPARISON_FTYPES = {336, 460}
REQUEST_TIMEOUT_SECONDS = 60
BATCH_SIZE = 300


def request_json(parameters: dict[str, object]) -> dict:
    """Make a form-encoded request to the USGS ArcGIS REST endpoint."""
    data = urlencode(parameters).encode("utf-8")
    request = Request(
        NHD_LAYER_URL,
        data=data,
        headers={"User-Agent": "Houston-DEM-NHD-comparison/1.0"},
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(f"USGS NHD service error: {result['error']}")
    return result


def download_nhd_flowlines(bounds_wgs84: tuple[float, float, float, float]) -> dict:
    """Download all NHD flowlines intersecting the DEM extent."""
    xmin, ymin, xmax, ymax = bounds_wgs84
    query = {
        "where": "1=1",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "returnIdsOnly": "true",
        "f": "json",
    }
    id_response = request_json(query)
    object_ids = id_response.get("objectIds")
    if object_ids is None:
        raise RuntimeError("USGS NHD response did not contain object IDs.")
    if not object_ids:
        raise RuntimeError("USGS NHD returned no flowlines for the DEM extent.")

    features: list[dict] = []
    for start in range(0, len(object_ids), BATCH_SIZE):
        batch_ids = object_ids[start : start + BATCH_SIZE]
        feature_response = request_json(
            {
                "objectIds": ",".join(str(object_id) for object_id in batch_ids),
                "outFields": "OBJECTID,GNIS_NAME,FTYPE,FCODE",
                "returnGeometry": "true",
                "outSR": 4326,
                "f": "geojson",
            }
        )
        batch_features = feature_response.get("features")
        if batch_features is None:
            raise RuntimeError("USGS NHD response did not contain GeoJSON features.")
        features.extend(batch_features)

    if len(features) != len(object_ids):
        raise RuntimeError(
            f"USGS NHD returned {len(features)} of {len(object_ids)} requested "
            "flowlines; refusing to write an incomplete comparison."
        )

    collection = {
        "type": "FeatureCollection",
        "name": "USGS NHD Flowline - Large Scale",
        "features": features,
    }
    NHD_PATH.write_text(json.dumps(collection), encoding="utf-8")
    return collection


def feature_type(feature: dict) -> int | None:
    value = feature.get("properties", {}).get("ftype")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def rasterize_flowlines(
    features: list[dict],
    out_shape: tuple[int, int],
    transform,
    source_crs,
) -> tuple[np.ndarray, list[list[tuple[float, float]]], int]:
    """Rasterize stream/river and canal/ditch lines and prepare map geometry."""
    raster_shapes = []
    map_segments = []
    selected_count = 0

    for feature in features:
        if feature_type(feature) not in COMPARISON_FTYPES:
            continue
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        projected = transform_geom("EPSG:4326", source_crs, geometry)
        raster_shapes.append((projected, 1))
        selected_count += 1

        geometry_type = projected["type"]
        coordinates = projected["coordinates"]
        if geometry_type == "LineString":
            map_segments.append(coordinates)
        elif geometry_type == "MultiLineString":
            map_segments.extend(coordinates)

    if not raster_shapes:
        raise RuntimeError(
            "No NHD Stream/River or Canal/Ditch features intersect the DEM extent."
        )

    nhd_mask = rasterize(
        raster_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    if not nhd_mask.any():
        raise RuntimeError("NHD linework could not be rasterized onto the DEM grid.")
    return nhd_mask, map_segments, selected_count


def score_networks(
    derived_mask: np.ndarray,
    nhd_mask: np.ndarray,
    cell_size_m: tuple[float, float],
) -> dict[int, tuple[float, float, float]]:
    """Measure how much of each rasterized network lies near the other."""
    if not derived_mask.any():
        raise RuntimeError("The derived stream network is empty at a threshold.")

    distance_to_derived = distance_transform_edt(
        ~derived_mask, sampling=cell_size_m
    )
    distance_to_nhd = distance_transform_edt(~nhd_mask, sampling=cell_size_m)
    scores = {}
    for tolerance_m in DISTANCE_TOLERANCES_M:
        nhd_match = float(
            np.mean(distance_to_derived[nhd_mask] <= tolerance_m) * 100
        )
        derived_match = float(
            np.mean(distance_to_nhd[derived_mask] <= tolerance_m) * 100
        )
        f1 = (
            2 * nhd_match * derived_match / (nhd_match + derived_match)
            if nhd_match + derived_match
            else 0.0
        )
        scores[tolerance_m] = (nhd_match, derived_match, f1)
    return scores


def read_raster_band(filename: Path, band_number: int) -> np.ma.MaskedArray:
    """Read one raster band from a file and return its masked pixel data."""
    with rasterio.open(filename) as dataset:
        return dataset.read(band_number, masked=True)


def plot_network_comparison(
    dem_band_data: np.ma.MaskedArray,
    extent: tuple[float, float, float, float],
    stream_masks: dict[int, np.ndarray],
    map_segments: list[list[tuple[float, float]]],
) -> None:
    """Plot a DEM band with the derived networks and NHD flowlines."""
    figure, axes = plt.subplots(1, len(STREAM_THRESHOLDS), figsize=(18, 7))
    for ax, threshold in zip(axes, STREAM_THRESHOLDS):
        ax.imshow(
            dem_band_data,
            extent=extent,
            origin="upper",
            cmap="Greys",
            alpha=0.38,
        )
        stream_layer = np.ma.masked_where(
            ~stream_masks[threshold], stream_masks[threshold]
        )
        ax.imshow(
            stream_layer,
            extent=extent,
            origin="upper",
            cmap=matplotlib.colors.ListedColormap(["#dc2626"]),
            vmin=0,
            vmax=1,
            alpha=0.8,
            interpolation="nearest",
            zorder=2,
        )
        ax.add_collection(
            LineCollection(
                map_segments,
                colors="#0077b6",
                linewidths=0.75,
                alpha=0.95,
                zorder=3,
            )
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_title(f"Accumulation threshold: {threshold:,} cells")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    figure.legend(
        handles=[
            Line2D([0], [0], color="#dc2626", linewidth=2, label="DEM-derived stream"),
            Line2D([0], [0], color="#0077b6", linewidth=1.5, label="NHD stream/river + canal/ditch"),
        ],
        loc="lower center",
        ncol=2,
        frameon=True,
    )
    figure.suptitle("Houston DEM-derived network compared with USGS NHD flowlines")
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    figure.savefig(MAP_PATH, dpi=220)
    plt.close(figure)


def write_summary(
    collection: dict,
    selected_count: int,
    scores_by_threshold: dict[int, dict[int, tuple[float, float, float]]],
) -> None:
    best_threshold = max(
        STREAM_THRESHOLDS,
        key=lambda threshold: scores_by_threshold[threshold][50][2],
    )
    lines = [
        "Houston DEM and USGS NHD flowline comparison",
        "",
        f"Retrieved (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Source: {NHD_LAYER_DESCRIPTION}",
        f"Service: {NHD_LAYER_URL}",
        f"Flowlines downloaded: {len(collection['features']):,}",
        (
            f"Comparison features: {selected_count:,} Stream/River (FTYPE 460) "
            "or Canal/Ditch (FTYPE 336)"
        ),
        (
            "Comparison method: both datasets were rasterized to the DEM's 25 m grid. "
            "Percentages are cell-based proximity measures, not surveyed line-length "
            "accuracy."
        ),
        "",
        (
            "Threshold | Tolerance (m) | NHD cells near DEM network (%) | "
            "DEM cells near NHD (%) | Balanced F1 (%)"
        ),
        "-" * 100,
    ]
    for threshold in STREAM_THRESHOLDS:
        for tolerance_m in DISTANCE_TOLERANCES_M:
            nhd_match, derived_match, f1 = scores_by_threshold[threshold][tolerance_m]
            lines.append(
                f"{threshold:>9,} | {tolerance_m:>13} | "
                f"{nhd_match:>28.1f} | {derived_match:>22.1f} | {f1:>16.1f}"
            )

    best_nhd, best_derived, best_f1 = scores_by_threshold[best_threshold][50]
    lines.extend(
        [
            "",
            (
                f"At a 50 m tolerance, threshold {best_threshold:,} has the highest "
                f"balanced cell-based match (F1 {best_f1:.1f}%; NHD-to-derived "
                f"{best_nhd:.1f}%, derived-to-NHD {best_derived:.1f}%). This is the "
                "closest of the tested thresholds by this specific metric, not a "
                "validation of channel locations."
            ),
            "",
            "Interpretation and limitations",
            (
                "Differences are expected: the accumulation threshold controls which "
                "headwater channels appear; the DEM is a 25 m resampled elevation "
                "surface; mapped NHD features and terrain-derived flow paths represent "
                "different data and mapping methods; and this rectangular DEM crop is "
                "not a watershed, so upstream contributing area is truncated at its "
                "edges. Urban drainage, canals, culverts, elevation artifacts, and "
                "mapping scale can also shift or add channels."
            ),
            (
                "NHD features are downloaded for the crop's intersecting extent, but "
                "proximity scores are calculated only for the Stream/River and "
                "Canal/Ditch feature types. Distances are measured between raster-cell "
                "centers, so the scores depend on the 25 m raster resolution and "
                "chosen tolerance."
            ),
            (
                "This is an exploratory alignment comparison. It does not estimate "
                "flow, flood extent, or flood risk."
            ),
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DEM_PATH.exists():
        raise FileNotFoundError(f"DEM not found: {DEM_PATH}")
    if not ACCUMULATION_PATH.exists():
        raise FileNotFoundError(
            f"Flow accumulation raster not found: {ACCUMULATION_PATH}. "
            "Run hydrology_analysis.py first."
        )

    with rasterio.open(DEM_PATH) as dem_source:
        dem_crs = dem_source.crs
        dem_transform = dem_source.transform
        dem_bounds = dem_source.bounds
        dem_shape = dem_source.shape
        cell_size_m = (abs(dem_source.res[1]), abs(dem_source.res[0]))
        bounds_wgs84 = transform_bounds(
            dem_crs, "EPSG:4326", *dem_bounds, densify_pts=21
        )
        extent = (
            dem_bounds.left,
            dem_bounds.right,
            dem_bounds.bottom,
            dem_bounds.top,
        )

    dem_band_data = read_raster_band(DEM_PATH, band_number=1)

    with rasterio.open(ACCUMULATION_PATH) as accumulation_source:
        if (
            accumulation_source.crs != dem_crs
            or accumulation_source.transform != dem_transform
            or accumulation_source.shape != dem_shape
        ):
            raise ValueError(
                "Flow accumulation raster must match the DEM CRS, grid, and shape."
            )
        accumulation = accumulation_source.read(1)
        valid = accumulation_source.read_masks(1) > 0

    collection = download_nhd_flowlines(bounds_wgs84)
    nhd_mask, map_segments, selected_count = rasterize_flowlines(
        collection["features"], dem_shape, dem_transform, dem_crs
    )
    stream_masks = {
        threshold: valid & (accumulation >= threshold)
        for threshold in STREAM_THRESHOLDS
    }
    scores_by_threshold = {
        threshold: score_networks(
            stream_masks[threshold], nhd_mask, cell_size_m
        )
        for threshold in STREAM_THRESHOLDS
    }

    plot_network_comparison(dem_band_data, extent, stream_masks, map_segments)
    write_summary(collection, selected_count, scores_by_threshold)
    print(f"Downloaded {len(collection['features']):,} NHD flowlines to {NHD_PATH}")
    print(f"Saved overlay map to {MAP_PATH}")
    print(f"Saved comparison metrics to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
