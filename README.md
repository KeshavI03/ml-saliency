# ml-saliency
This repo serves to present a few methods of saliency detection, specifically for the purpose of image segmentation to imitate short-fov cameras through selective blurring of the background.



## Analytical Method

- Through derivatives of gaussian pyramids, a contrast map is generated
- Given multiple scales, areas of low level and high level change are found
- We then create contours around these regions
- We next find closed contours and pick the best one (largest and closest)
- We dilate the selected contour, fill it, and use it as the mask on top of our original image

### Issues
- Areas of detail next to eachother often lead to connected contours that should be seperate
- This method does not take into account real saliency and the nuances of human vision
- It simply isn't very accurate the way it is currently implemented
- It is memory intensive, requiring multiple gaussian blurs of the same image

##  ML Method
- We train a UNet fully convolutional neural network
- Our inputs and output are an image and a segmentation of the same size
- After training, we can validate and use the system to segment images
- We can then use the map to create a blur mask and apply it

### Issues

- When the system fails, issue identification isn't possible given the ML system
