import splitfolders


# 0. transforming folder level data into training/validation/test folders
# creates 3 subdirectories with each % assigned .6, .2, .2
input_folder = "input_dataset"
output = "processed_data"
splitfolders.ratio(input_folder, output, seed=42, ratio=(.6, .2, .2))
