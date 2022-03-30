
filename=recording_with_rotation
limit_high=1.0
limit_low=0.9

state_transitions: label
	models/preprocess_dataset.py

label: threshold
	trajectory_extract/regression.py processed_data/thresh/*

threshold: smooth
	trajectory_extract/threshold.py processed_data/smoothed/*

smooth: split
	trajectory_extract/smoothing.py processed_data/*

split: regularize
	trajectory_extract/split.py $(limit_high) $(limit_low) processed_data/$(filename)-regularized.csv

regularize: extract
	trajectory_extract/regularize.py processed_data/$(filename).csv

extract: rawdata/$(filename).bag
	trajectory_extract/extract.py rawdata/$(filename).bag



