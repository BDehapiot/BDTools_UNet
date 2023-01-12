#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path

#%% Open data

raw = []; mask = []
dirlist = sorted(Path('data_RBCs', 'train').glob('*.tif'))
for path in dirlist:
    if 'mask' not in path.name: 
        raw.append(io.imread(path))
    else:
        mask.append(io.imread(path))

raw = np.stack(raw)
mask = np.stack(mask)

#%% Prepare data 

# 0 to 1 normalization
for img in raw:
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

#%% Display

viewer = napari.Viewer()
viewer.add_image(raw)






