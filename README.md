# GIST-feature-extractor

[UnOfficial]
This repository is a reimplementation Matlab code implemented in this paper [Modeling the shape of the scene: a holistic representation of the spatial envelope](http://people.csail.mit.edu/torralba/code/spatialenvelope/) in Python.

__Note : Images that can be used in this repository are only grayscale images. (W × H × 1)__

![top-page](https://raw.githubusercontent.com/imoken1122/GIST-feature-extractor/img/explain.png)


## Prerequisites
The folloing liblaries are required to be installed 
- Python 3.6
- numpy
- pillow
- feather-format

## Usage
  
__1. Installation__

```
$ git clone https://github.com/imoken1122/GIST-feature-extractor.git
```

<br>

__2. Use GIST-feature extractor__
  

Change these parameters if necessary.
```python:main.py
param = {
        "orientationsPerScale":np.array([8,8]),
         "numberBlocks":[10,10],
        "fc_prefilt":10,
        "boundaryExtension": 10
        
}
```
<br>

Specifies an image name or a folder path containing several images and output path saving extracted gist-feature as option (__Extention of output file is ".feather"__)
```
--input_path <your path>
--output_path <your path>

```
<br>
Let's extracting GIST-feature !!  
The following is an example.

```sh
$ python main.py --input_path image_name.png --output_path gist.feather
```
or

```sh
$ python main.py --input_path image_list --output_path gist.feather
```


