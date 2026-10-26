"""Deterministic evaluation contract adapted from the image benchmark trainers."""
import numpy as np
from PIL import Image, ImageOps

def deterministic_grid_boxes(width, height, fraction=0.45, count=9):
    if width < 1 or height < 1 or not 0 < fraction <= 1 or count not in range(1,10):
        raise ValueError('Positive dimensions, fraction in (0,1], and 1..9 crops required')
    side = max(1,int(round(min(width,height)*fraction)))
    return [(int(round(x*(width-side))),int(round(y*(height-side))),side)
            for y in (0,0.5,1) for x in (0,0.5,1)][:count]

def evaluation_crops(path, *, size=384, fraction=0.45, count=9):
    if size < 1:
        raise ValueError('Image size must be positive')
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        return [image.crop((x,y,x+s,y+s)).resize((size,size),Image.Resampling.BICUBIC)
                for x,y,s in deterministic_grid_boxes(*image.size,fraction,count)]

def normalized_array(image, *, mean, std):
    mean,std = np.asarray(mean,dtype=float),np.asarray(std,dtype=float)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Three finite channel means and positive standard deviations required')
    return ((np.asarray(image.convert('RGB'),dtype=np.float32)/255-mean)/std).transpose(2,0,1)
