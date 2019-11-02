from GIST import _gist_extract
from PIL import Image
import numpy as np
param = {
        "orientationsPerScale":np.array([8,8]),
         "numberBlocks":4,
        "fc_prefilt":4,
        "boundaryExtension": 20
        
}
path = ""
img = np.array(Image.open(path).convert("L"))
gist = _gist_extract(img, param)
print(gist)