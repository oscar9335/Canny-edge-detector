import cv2 
import numpy as np
from scipy import ndimage


# Noise reduction using Gaussian blur 
# kernel size using 5x5
def gaussian_blur(size, sigma=1):
    size = int(size) // 2
    # create 2D numpy array x,y  
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    # the equation
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def sobel2(image):
    # Sobel kernels
    kernel_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]])

    kernel_y = np.array([[-3, -10, -3],
                         [0, 0, 0],
                         [3, 10, 3]])

    # Convolve the kernels with the image
    gradient_x = ndimage.filters.convolve(image, kernel_x)
    gradient_y = ndimage.filters.convolve(image, kernel_y)

    cv2.imshow('virtical Gradient', cv2.convertScaleAbs(gradient_x))
    cv2.imshow('horizontal Gradient', cv2.convertScaleAbs(gradient_y))

    # Combine gradients
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    gradient_magnitude = np.hypot(gradient_y, gradient_x)

    # normalize and convert type
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    return gradient_magnitude, gradient_direction

def non_max_suppression(gradient_magnitude, gradient_direction):
    # M corresponds to the height or number of rows, and N corresponds to the width or number of columns of the gradient_magnitude array.
    M, N = gradient_magnitude.shape
    suppressed = np.zeros((M, N), dtype=np.uint8)

    angle = np.round(gradient_direction / (np.pi/4)) % 4

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            mag = gradient_magnitude[i, j]
            if angle[i, j] == 0:  # Angle is 0 (horizontal)
                neighbors = (mag, gradient_magnitude[i, j-1], gradient_magnitude[i, j+1])

            elif angle[i, j] == 1:  # Angle is 45 degrees
                neighbors = (mag, gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1])

            elif angle[i, j] == 2:  # Angle is 90 degrees (vertical)
                neighbors = (mag, gradient_magnitude[i-1, j], gradient_magnitude[i+1, j])

            elif angle[i, j] == 3:  # Angle is 135 degrees
                neighbors = (mag, gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1])

            if mag >= max(neighbors):
                suppressed[i, j] = mag
    
    return suppressed

def double_threshold(gradient_magnitude, low_threshold_ratio=0.05, high_threshold_ratio=0.10):

    high_threshold = gradient_magnitude.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    M, N = gradient_magnitude.shape
    res = np.zeros((M,N), dtype=np.uint8)

    strong_i, strong_j = np.where(gradient_magnitude >= high_threshold)
    weak_i, weak_j = np.where((gradient_magnitude <= high_threshold) & (gradient_magnitude >= low_threshold))
    
        
    weak = np.int32(25)
    strong = np.int32(255)

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res, strong, weak

def hysteresis(gradient_magnitude, weak, strong=255):
    M, N = gradient_magnitude.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (gradient_magnitude[i,j] == weak):
                try:
                    if ((gradient_magnitude[i+1, j-1] == strong) or (gradient_magnitude[i+1, j] == strong) or (gradient_magnitude[i+1, j+1] == strong)
                        or (gradient_magnitude[i, j-1] == strong) or (gradient_magnitude[i, j+1] == strong)
                        or (gradient_magnitude[i-1, j-1] == strong) or (gradient_magnitude[i-1, j] == strong) or (gradient_magnitude[i-1, j+1] == strong)):
                        gradient_magnitude[i, j] = strong
                    else:
                        gradient_magnitude[i, j] = 0
                except IndexError as e:
                    pass
    return gradient_magnitude

def main():
    gray_guitar = cv2.imread("guitar.jpg",cv2.IMREAD_GRAYSCALE)
    gray_house = cv2.imread("house.jpg",cv2.IMREAD_GRAYSCALE)
    gray_lenna = cv2.imread("lenna.jpg",cv2.IMREAD_GRAYSCALE)

    # Display the original and 
    cv2.imshow('Original Image', gray_lenna)

# Step 1. apply gaussian filtwer
    # Apply the Gaussian kernel to the image
    gaussian_kernel = gaussian_blur(3,1)
    image = cv2.filter2D(gray_lenna, -1, gaussian_kernel)
    #Gaussian filtered images
    cv2.imshow('Gaussian Filtered Image', image)

# Step 2. apply Sobel operator to Calculation Gradient
# gradient_magnitude = sqrt(Ix**2 + Iy**2)
# gradient_direction = direction = arctan(Iy/Ix)
    # Apply Sobel operator
    gradient_magnitude, gradient_direction = sobel2(image)
    # Display Sobel operator
    cv2.imshow('Gradient Magnitude', cv2.convertScaleAbs(gradient_magnitude))

# Step 3. non-maximum suppression
# check if the the selected pixel value is the most intense values of the same direction 2 neighbors pixels 
    # Apply non-maximum suppression
    suppressed_image = non_max_suppression(gradient_magnitude, gradient_direction)
    # Display non-maximum suppression
    cv2.imshow("Non-maximum suppressed image", suppressed_image)

# Step 4. double threshold
# classify pixels into 3 kind: strong(final edge), weak(between strong and non-relevent),  non-relevent(not used as edge)
    res, strong, weak = double_threshold(gradient_magnitude)
    cv2.imshow("double threshold", res)

# Step 5. Edge Tracking by Hysteresis 
# ransforming weak pixels into strong ones, if and only if at least one of the pixels around the one being processed is a strong one
    last_result = hysteresis(gradient_magnitude, weak=25, strong=255)
    cv2.imshow("hysteresis", last_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()


