# GIST-feature-extractor
This repository is a reimplementation Matlab code implemented in this paper [Modeling the shape of the scene: a holistic representation of the spatial envelope](http://people.csail.mit.edu/torralba/code/spatialenvelope/) in Python.

__Images that can be used in this repository are still only grayscale images. (W × H × 1)__

## Prerequisites
The folloing liblaries are required to be installed 
- Python 3.6
- Numpy
- pillow

## Usage

__1. Installation__
```
$ git clone https://github.com/imoken1122/GIST-feature-extractor.git
```

__2. Use GIST-feature extractor__

Change these parameters if necessary and specify the path where the image is located.
```python
param = {
        "orientationsPerScale":np.array([8,8,8]),
         "numberBlocks":10,
        "fc_prefilt":10,
        "boundaryExtension": 10
        
}
```

Specifies an image name or a folder path containing several images as command line argument and output path saving extracted gist-feature. (__Extention of output file is feather__)
  
Let's extracting GIST-feature !!  
The following is an example.

```
$ python main.py image_name.png feature_list/gist.feather
```
or

```
$ python main.py image_list feature_list/gist.feather
```


