# Canny-edge-detector

# gaussian_blur
this would fdo the gaussian blur, using this function ![alt text](image.png)

# sobel2
this apply the 2 sobel kernel on the given image and calculate the gradient_magnitude and direction

# non_max_suppression
using the direction to caluculate the nighbors of selected pixels, if they are obvious then the selected pixel

# double_threshold
classify the pixels into 3 type, strong, weak, non, the strong is obvious the edge, and weak is possible, no is not

# hysteresis
transforming weak pixels into strong ones