### What Synthseg is doing in [labels_to_image_model.py](https://github.com/BBillot/SynthSeg/blob/master/SynthSeg/labels_to_image_model.py) ###

- reformat resolutions (?)  mysterious utils function
- get shapes                mysterious utils function, not sure why this is needed given that resolution is fixed
- deforming:                layers.RandomSpatialDeformation
- cropping:                 layers.RandomCrop
- flipping:                 layers.RandomFlip
- make synthetic image:     layers.SampleConditionalGMM
- apply bias field          layers.BiasFieldCorruption
- intensity augmentation:   layers.IntensityAugmentation
- loop over channels (?)    not sure why if shape has channel dim 0 (256,256,256) 
    - (ei) randomize resolution
    - (or) blurring and downsampling (?)
- concatenate all channels back
- compute image gradients (?)
    - layers.ImageGradients
    - layers.IntensityAugmentation
- map gen lab to seg        layers.ConvertLabels
- make brain model, which just wraps the image and label lambdas into a class

> this is only done once. Afterwards image generation is done by calling brain model itself
> which is just two KL.lambdas


### Notes ###

- They only really used KL for the Lambda in the main labels_to_image_model file
- There is 2000 lines of mysterious python (mixed plain python, numpy and tf) in ext.layers
- They are huge fans of list comprehensions. This would explain why things are slow
Example:

```
tmp_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in output_shape]
```
Calls:
```
def find_closest_number_divisible_by_m(n, m, answer_type='lower'):
    """Return the closest integer to n that is divisible by m. answer_type can either be 'closer', 'lower' (only returns
    values lower than n), or 'higher' (only returns values higher than m)."""
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == 'lower':
            return lower
        elif answer_type == 'higher':
            return higher
        elif answer_type == 'closer':
            return lower if (n - lower) < (higher - n) else higher
        else:
            raise Exception('answer_type should be lower, higher, or closer, had : %s' % answer_type)
```

> this is the slowest this code could possibly be I think.

Whereas they could have done this entirely in numpy, perhaps even faster if the np.arange(n+1) 
was replaced by an existing vector in the function declaration

```
import numpy as np

def find_closest_number_divisible_by_m_vectorized(n, m):
    # Create an array of numbers from 0 to n
    numbers = np.arange(n+1)
    
    # Find the remainder of each number when divided by m
    remainders = numbers % m
    
    # Find the indices of the numbers that are divisible by m
    divisible_indices = np.where(remainders == 0)[0]
    
    # Return the largest number that is divisible by m
```
This also works in torch:
```
import torch

def find_closest_number_divisible_by_m_torch(n, m):
    # Create a tensor of numbers from 0 to n
    numbers = torch.arange(n+1)
    
    # Find the remainder of each number when divided by m
    remainders = numbers % m
    
    # Find the indices of the numbers that are divisible by m
    divisible_indices = torch.where(remainders == 0)[0]
    
    # Return the largest number that is divisible by m
    return numbers[divisible_indices[-1]]
```

### Conclusion ###

This would probably take about 2 weeks, if I'm smart.
This might also take 1 afternoon, if Claude is just AGI