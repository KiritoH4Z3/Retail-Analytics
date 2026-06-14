# Retail Zone Analytics

A computer-vision pipeline that tracks shoppers through a store, measures how long they dwell in each zone, and surfaces the results in a manager-facing dashboard.

## Results

- Detects and tracks people in retail video using YOLOv8 + DeepSORT, then assigns each tracked person to one of **5 defined store zones** (Entrance, Checkout Counter, Center Aisle, Back Aisle, Right Aisle) via polygon point-in-region tests.
- Logs every zone visit as a dwell event (track ID, zone, entry/exit timestamps, dwell seconds) to a SQLite database, filtering out transits shorter than 1 second.
- Streamlit dashboard reports KPIs and analytics derived from the logged data:
  - Total unique visitors, total zone events, average dwell time, most-visited zone
  - Unique visitors and average dwell time per zone
  - A 0-100 **zone engagement score** combining normalized visitor count and dwell time
  - Zone visit frequency, a ranked dwell-time breakdown table, and per-customer zone journeys
  - Automated **manager alerts** for low-engagement zones, short dwell times, dead (zero-activity) zones, and checkout queue bottlenecks

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![DeepSORT](https://img.shields.io/badge/DeepSORT-tracking-555555)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)

## Architecture

`main.py` runs the video pipeline: YOLOv8 detects people, DeepSORT assigns persistent track IDs, and each track's center point is tested against predefined zone polygons to determine the current zone. Entry and exit times per zone are tracked in memory and written to a SQLite `dwell_events` table when a person leaves a zone or their track is lost. `dashboard.py` reads that database and renders the analytics, engagement scores, and manager alerts with Streamlit and Plotly.

## How to Run

```bash
git clone https://github.com/KiritoH4Z3/Retail-Analytics.git
pip install -r requirements.txt

# 1. Process video and log dwell events to the SQLite database
python main.py

# 2. Launch the analytics dashboard
streamlit run dashboard.py
```

> Note: `main.py` and `dashboard.py` reference hardcoded Windows paths for the input video and `dwell_events.db`. Update `DB_PATH` and the `cv2.VideoCapture(...)` source to match your local video file and database location before running.

## About

Built by Abdullah Mohammed Hazeq as a computer-vision retail analytics project, turning in-store video into actionable shopper-behavior and zone-performance insights.
