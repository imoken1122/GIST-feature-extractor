from model.GIST import GIST
from tqdm import tqdm
import sys
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import re
import feather
import os
import multiprocessing as mp


param = {
        "orientationsPerScale":np.array([8,8]),
         "numberBlocks":[10,10],
        "fc_prefilt":10,
        "boundaryExtension":32
}
class Dataloader():
	def __init__(self,input_path, output_path):
		self.input_path = input_path
		self.output_path = output_path
		self.is_dir = 0 if re.search("\.",input_path) != None else 1

	def get_inputfile(self) -> list:
		if self.is_dir:
			# dirctory in images
			path = f"./{self.input_path}/"
			a = sorted(os.listdir(path))
			file_list = list(map(lambda x: path + x, a))

			return file_list
		else:
			# image file such png, jpg etc..
			path = f"./{self.input_path}"
			return [path]
	def save_feature(self,x:np.array):
		if self.is_dir:
			gist_df = pd.DataFrame(x, columns = [f"gist_{i}" for i in range(x.shape[1])])
		else:
			gist_df = pd.DataFrame(x.reshape(1,-1), columns = [f"gist_{i}" for i in range(x.shape[1])])

		gist_df.to_feather(f"./{self.output_path}")	

def _get_gist(param:dict,file_list:list) -> np.array:
	img_list = list(map(lambda f :np.array(Image.open(f).convert("L")), file_list))
	gist = GIST(param)

	with mp.Pool(mp.cpu_count()) as pool:
		p = pool.imap(gist._gist_extract,img_list[:])
		gist_feature = list(tqdm(p, total = len(img_list)))
	return np.array(gist_feature)


if __name__ == "__main__":
	arg = argparse.ArgumentParser()
	arg.add_argument("--input_path",default = "image_list/")
	arg.add_argument("--output_path",default= "gist.feather")
	arg.add_argument("--save",default=True)
	args = arg.parse_args()
	print(args)

	data = Dataloader(args.input_path,args.output_path)
	file_list = data.get_inputfile()
	gist_feature = _get_gist(param,file_list)
	print(gist_feature.shape)
	if args.save == True:
		data.save_feature(gist_feature)