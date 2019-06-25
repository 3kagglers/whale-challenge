"""Main file for whale project."""

import numpy as np
import pandas as pd

# =========================================================================== #
# loading and checking data
# =========================================================================== #

# Load data
# Probably use pd.read_csv or Preprocessor class

# Detect outliers and remove them
# Preprocessor class has a remove_outliers method for this

# Load both test and train datasets
# Keep track of size of train dataset
# Merge them
# Apply operations to both datasets

# =========================================================================== #
# Feature analysis
# =========================================================================== #

# Identify tail

# Crop image down to ROI

# Remove texts and unwanted features

# Train classifier to tell good crops from bad ones

# Get only tail

# =========================================================================== #
# Filling missing values
# =========================================================================== #

# Run classifier on images to make sure that we have a tail in all

# =========================================================================== #
# Feature engineering
# =========================================================================== #

# Check brainstorming for whale data analysis card in documents list for ideas

# =========================================================================== #
# Changing feature representation
# =========================================================================== #

# Apply filters and what not, extrapolation of feature engineering

# Figure out a way for identifying new whales

# =========================================================================== #
# Modeling
# =========================================================================== #

# Define model(s), CNN, SOM,...
# Train model(s)
# Plot training data
# Ckeck training data available (cross-validation?)
# Tune hyper-parameters, strategies for this are e.g. gridSearch or genetic alg

# =========================================================================== #
# Prediction
# =========================================================================== #

# Test it out
