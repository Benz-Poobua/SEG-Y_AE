import os
import re
import logging
import segyio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from segysak.segy import segy_header_scan
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define paths
root: Path = Path.cwd().parent
data_path: Path = root / 'data'
processed_data_path: Path = root / 'processed_data'

# Ensure processed_data_path exists
os.makedirs(processed_data_path, exist_ok=True)

def scan_segy_headers(segy_file: str) -> pd.DataFrame:
    """
    Scan the headers of a 2D SEG-Y file and return the first 100 rows.
    """
    logging.info(f"Scanning headers for SEG-Y file: {segy_file}")
    scan = segy_header_scan(segy_file)
    return scan

def parse_trace_headers(segyfile: segyio.SegyFile) -> pd.DataFrame:
    """
    Extract selected SEG-Y trace header fields into a pandas DataFrame.
    """
    logging.info("Starting trace header extraction.")
    all_headers = segyio.tracefield.keys

    useful_keys = [
        "TraceNumber", "FieldRecord", "CDP", "CDP_X", "CDP_Y",
        "SourceX", "SourceY", "GroupX", "GroupY", "offset",
        "TRACE_SAMPLE_COUNT", "TRACE_SAMPLE_INTERVAL",
        "DelayRecordingTime", "MuteTimeStart", "MuteTimeEND",
        "SourceGroupScalar", "CoordinateUnits", "GainType",
        "TotalStaticApplied", "YearDataRecorded", "DayOfYear"
    ]

    n_traces = segyfile.tracecount
    logging.info(f"Number of traces detected: {n_traces}")
    df = pd.DataFrame(index=range(n_traces))

    for key in useful_keys:
        if key in all_headers:
            try:
                df[key] = segyfile.attributes(all_headers[key])[:]
                logging.debug(f"Successfully read header '{key}'")
            except Exception as e:
                logging.warning(f"Could not read header '{key}': {e}")
                df[key] = None
        else:
            logging.warning(f"Header '{key}' not found in SEG-Y tracefield map.")
            df[key] = None

    logging.info("Trace header extraction completed.")
    return df

def parse_text_header(segyfile: segyio.SegyFile) -> Dict[str, str]:
    """
    Parse the textual (EBCDIC) header of the SEG-Y file.
    """
    logging.info("Parsing textual (EBCDIC) header.")

    try:
        raw_text = segyio.tools.wrap(segyfile.text[0])
    except Exception as e:
        logging.error(f"Failed to read text header: {e}")
        return {}

    header_lines = re.split(r'C ', raw_text)[1:]
    header_lines = [line.replace('\n', ' ').strip() for line in header_lines]
    formatted_header = {f"C{str(i+1).zfill(2)}": line for i, line in enumerate(header_lines)}

    logging.info("Text header parsed successfully.")
    return formatted_header

def plot_segy(segyfile: str, clip_percentile: float =99, cmap: str ='RdBu') -> None:
    """
    Plot a 2D seismic SEG-Y line using matplotlib.
    """
    logging.info(f"Opening SEG-Y file for plotting: {segyfile}")

    with segyio.open(segyfile, ignore_geometry=True) as f:
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000  # in milliseconds
        n_samples = f.samples.size
        twt = f.samples  # Two-way travel time [ms]
        data = f.trace.raw[:]  # numpy array of shape (n_traces, n_samples)

    logging.info(f"Loaded {n_traces} traces with {n_samples} samples each. Sample rate: {sample_rate} ms.")
    vm = np.percentile(data, clip_percentile)
    logging.info(f"Clipping amplitude values at ±{vm:.2f} based on {clip_percentile}th percentile.")

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1)
    extent = [1, n_traces, twt[-1], twt[0]]  # seismic convention: time axis reversed
    im = ax.imshow(data.T, cmap=cmap, vmin=-vm, vmax=vm, aspect='auto', extent=extent)

    ax.set_xlabel('CDP number (Trace number)')
    ax.set_ylabel('Two-Way Travel Time (ms)')
    ax.set_title(f'Seismic Line: {Path(segyfile).name}')
    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    plt.show()

    logging.info("Seismic plot generated successfully.")

def cut_segy(input_filename: str, output_filename: str, cut_time_ms: int) -> None:
    """
    Cut a SEG-Y file to a maximum two-way travel time and save it.
    """
    input_path = data_path / input_filename
    output_path = processed_data_path / output_filename

    logging.info(f"Cutting SEG-Y file at {cut_time_ms} ms.")
    logging.info(f"Input: {input_path} → Output: {output_path}")

    with segyio.open(input_path, ignore_geometry=True) as src:
        sample_rate = segyio.tools.dt(src) / 1000  # microseconds to milliseconds
        cut_sample = int(cut_time_ms / sample_rate)
        spec = segyio.tools.metadata(src)
        spec.samples = spec.samples[:cut_sample]

        with segyio.create(output_path, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.bin.update(hns=cut_sample)

            for i in range(src.tracecount):
                dst.header[i] = src.header[i]
                dst.trace[i] = src.trace[i][:cut_sample]

    logging.info(f"Cut SEG-Y file saved as: {output_filename}")

def convert_to_numpy(segy_filename: str, numpy_filename: str) -> np.ndarray:
    """
    Convert a cut SEG-Y file to a NumPy array and save it as a .npy file.
    """
    input_path = processed_data_path / segy_filename
    output_path = processed_data_path / numpy_filename

    logging.info(f"Reading SEG-Y from: {input_path}")
    try:
        with segyio.open(str(input_path), ignore_geometry=True) as f:
            seismic_data_numpy = f.trace.raw[:]  # shape: (n_traces, n_samples)

        logging.info(f"NumPy shape: {seismic_data_numpy.shape}")
        np.save(output_path, seismic_data_numpy)
        logging.info(f"Saved NumPy array: {output_path}")
        return seismic_data_numpy

    except Exception as e:
        logging.error(f"Failed SEG-Y to NumPy conversion: {e}")
        raise

if __name__ == '__main__':
    logging.info("This is a utility module. Use these functions in your main script.")