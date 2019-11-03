from GIST import GIST
from tqdm import tqdm
import sys
from PIL import Image
import numpy as np
import pandas as pd
import re
import feather
import os
import multiprocessing as mp
param = {
        "orientationsPerScale":np.array([8,8,8]),
         "numberBlocks":10,
        "fc_prefilt":10,
        "boundaryExtension": 10
}
def _get_filelist(file_name, is_dir = True):
	if is_dir:
		# dirctory in images
		path = f"./image_list/{file_name}/"
		a = sorted(os.listdir(path))
		file_list = list(map(lambda x: path + x, a))

		return file_list
	else:
		# image file such png, jpg etc..
		path = f"./{file_name}"
		return [path]

def _get_gist(param,file_list):
	img_list = list(map(lambda f :np.array(Image.open(f).convert("L")), file_list))
	gist = GIST(param)

	with mp.Pool(mp.cpu_count()) as pool:
		p = pool.imap(gist._gist_extract,img_list)
		gist_feature = list(tqdm(p, total = len(img_list)))
	return np.array(gist_feature[0])

def main(file_name):
	is_dir = 0 if re.search("\.",file_name) != None else 1

	file_list = _get_filelist(file_name, is_dir = is_dir)
	gist_feature = _get_gist(param,file_list)
	print(gist_feature.shape)
	gist_df = pd.DataFrame(np.array(gist_feature), columns = [f"gist_{i}" for i in range(len(gist_feature))])
	gist_df.to_feather("output path")

if __name__ == "__main__":
	arg = sys.argv
	
	main(arg[1])
