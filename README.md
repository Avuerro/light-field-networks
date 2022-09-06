# light-field-networks

A repo dedicated to the Light Field Network model I worked with during my thesis. 


## Novel view synthesis
You can use the original LFN (since that one works best) to synthesize novel views, see `synthesizing_novel_views.ipynb` for instructions/examples.

## Geometry.
The light field network is in essence a neural network which parameterizes the light field function for a scene.
Since the light field function assigns to each ray intersecting a scene the color, we can construct depth maps and point clouds using this function in combination with the intersection points.

In `extracting_geometry.ipynb` you will find examples on how to run it. It should be able to run after downloading the data and weights and modifying some indicated parameters.

### Note
This is still a work in progress in the sense that all code works, however, I converted my working repo to this repo, so some parts might be missing. 
In case of errors, bugs or any other type of feedback feel free to reach out.


Most of the code in this repo is based on or copied from the impressive work done by Sitzmann et al. and can be found in the [original repo on light field networks](https://github.com/vsitzmann/light-field-networks).