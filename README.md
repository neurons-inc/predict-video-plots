# Generating video plots for Predict

This repository contains a Python script for generating video montage of Predict video heatmaps and
their corresponding graph for visualising the variation in Cognitive Demand and Focus.

## Requirements

Please install all required Python libraries and packages:

```
apt-get update
apt-get install -y ffmpeg
pip install --upgrade pip
pip install -r requirements.txt
```

## Guide

Please execute the script as follows:

`python gen_video_plots.py --path_video_heatmap <video-total-attention-heatmap-mp4> --path_scores_csv <video-results-frame-by-frame-csv> --metric both`

The command line arguments are as follows:

* --path_video_heatmap - Path to Predict video heatmap as obtained from raw downloads.
* --path_scores_csv - Path to Predict scores (frame-level) CSV file.
* --metric - Optional argument (default: "both") to generate only Focus graph (set "focus" here) or only Cognitive Demand graph (set "cognitive_demand" here). If not specified, both metrics will be displayed.