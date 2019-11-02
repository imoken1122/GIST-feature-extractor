# GIST-feature
This repository is a reimplementation Matlab code implemented in this paper [Modeling the shape of the scene: a holistic representation of the spatial envelope](http://people.csail.mit.edu/torralba/code/spatialenvelope/) in Python.

__Images that can be used in this repository are only grayscale images. (W × H × 1)__

## Prerequisites
The folloing liblaries are required to be installed 
- Python 3.6
- Numpy
- pillow

## Usage

__1. Installation__
```
$ git clone https://github.com/imoken1122/GIST-feature.git
```

__2. Use GIST-feature extractor__

Change these parameters if necessary and Specify the path where the image is located
```python
param = {
        "orientationsPerScale":np.array([8,8]),
         "numberBlocks":4,
        "fc_prefilt":4,
        "boundaryExtension": 20
        
}
```
```python
path = f"./image/{...}"
```

Let's extracting GIST-feature.

```
$ python main.py
```

