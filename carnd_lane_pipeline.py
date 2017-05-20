# This file contains utility functions for the Carnd-Project Advanced lane findings
# Author: Luca Fiaschi, luca.fiaschi@gmail.com

import cv2
import pickle
import numpy as np

# Load the camera calibration specs which were produced from the notebook Camera_Calibration.ipynb
CAMERA = pickle.load(open('./calibrarion.p','rb'))


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def undistort(img):
    """
    Undistort an image using the camera specs
    :param img: 
    :return: 
    """
    undist = cv2.undistort(img, CAMERA['mtx'], CAMERA['dist'], None, CAMERA['mtx'])
    return undist


def gaussian_blur(img, sigma):
    """
    Applies a Gaussian kernel to smooth the image
    """
    kernel_size = 11
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255, ksize= 7):

    """
    Apply a treshold based on the absolute value of the sobel filter in a certain direction.
    Sobel derivative image is rescaled to 8bit hence the treshold should be in [0, 255]
    :param gray: grayscale image
    :param orient: orientation for the sobel operator
    :param thresh_min: 
    :param thresh_max:
    :param ksize: 
    :return: 
    """

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize= ksize))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize= ksize))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def mag_thresh(gray, thresh_min=0, thresh_max = 255, ksize= 7):

    """
    Apply a treshold based on the absolute value of the sobel filter in a certain direction.
    Sobel derivative image is rescaled to 8bit hence the treshold should be in [0, 255]
    :param gray: grayscale image
    :param thresh_min: 
    :param thresh_max: 
    :param ksize: 
    :return: 
    """

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)

    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1

    return binary_output


def dir_threshold(gray, thresh_min=0, thresh_max=np.pi / 2, ksize=7):
    """
    Calculate the direction of the image gradient and takes a trheshold [0, Pi/2].
    :param gray: 
    :param thresh_min: 
    :param thresh_max: 
    :param ksize: 
    :return: 
    """

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1

    return binary_output


def segment(img):
    """
    Main function to segment the lane lines:
    :param img: RGB image
    :return: Binary segmented image with the lane lines
    """

    # 1. Gaussian Blur
    img = img.copy()
    img = gaussian_blur(img, 2)

    # 2. Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]

    # 3. Threshold o the image gradient
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', thresh_min=10, thresh_max=255)
    grady = abs_sobel_thresh(gray, orient='y', thresh_min=60, thresh_max=255)
    mag_binary = mag_thresh(gray, thresh_min=60, thresh_max=255)
    dir_binary = dir_threshold(gray, thresh_min=.65, thresh_max= 1.05)

    # Combine all the thresholded images for the gradients
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # 4. Threshold color channel
    s_binary = np.zeros_like(combined)
    s_binary[(s > 120) & (s < 255)] = 1

    # Combines color and gradient binary images
    color_binary = np.zeros_like(combined)
    color_binary[(s_binary > 0) | (combined > 0)] = 1

    # Define the region parameters taken from https://github.com/wonjunee/Advanced-Lane-Finding

    imshape = img.shape
    left_bottom = (100, imshape[0])
    right_bottom = (imshape[1] - 20, imshape[0])
    apex1 = (610, 410)
    apex2 = (680, 410)

    inner_left_bottom = (310, imshape[0])
    inner_right_bottom = (1150, imshape[0])
    inner_apex1 = (700, 480)
    inner_apex2 = (650, 480)

    vertices = np.array([[left_bottom, apex1, apex2, \
                          right_bottom, inner_right_bottom, \
                          inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)
    # Masked area
    color_binary = region_of_interest(color_binary, vertices)

    return color_binary


def corners_unwarp(img):

    # Define the region parameters taken from https://github.com/wonjunee/Advanced-Lane-Finding

    roi_source = [[150 + 430, 460], [1150 - 440, 460], [1150, 720], [150, 720]]

    # Choose an offset from image corners to plot detected corners
    offset1 = 200  # offset for dst points x value
    offset2 = 0  # offset for dst points bottom y value
    offset3 = 0  # offset for dst points top y value
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    # For source points I'm grabbing the outer four detected corners
    src = np.float32(roi_source)
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    dst = np.float32([[offset1, offset3],
                      [img_size[0] - offset1, offset3],
                      [img_size[0] - offset1, img_size[1] - offset2],
                      [offset1, img_size[1] - offset2]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M, Minv




class Lane(object):
    """
    An Helper Class to keep track of the detected Lanes
    """

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []

        self.recent_fits = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # x values in windows
        self.windows = np.ones((3, 12)) * -1


def find_window_centroids(binary_warped, nwindows=9, margin=100):
    """
    Implementation provided by Udacity of lane search with sliding windows
    :param binary_warped: binary warped image with segmented lane lines
    :param nwindows: 
    :param margin: 
    :return: 
    """

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fit, right_fit, ploty, left_fitx, right_fitx


def search_around_previous_fit(binary_warped, left_fit, right_fit):
    """
    Implement Lane search strategy by only searching the region around the previous lane
    :param binary_warped: 
    :param left_fit: 
    :param right_fit: 
    :return: 
    """

    # Implementation provided by Udacity

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit, right_fit, ploty, left_fitx, right_fitx


class LaneFinder(object):
    """  
    Main class for the project, takes care to fit the lanes to a series of images,
    keep track of the position of the lanes across several images
    
    This class is suppose to process an image after the other through the process image function
    
    """
    def __init__(self):



        self.left_lane = Lane()
        self.right_lane = Lane()

    @staticmethod
    def get_curvature(y, fitx):

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(y)
        # Define conversions in x and y from pixels space to meters
        # assume the lane is about 30 meters long and 3.7 meters wide
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        fit_cr = np.polyfit(y * ym_per_pix, fitx * xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        return curverad

    @staticmethod
    def get_position(pts, image_shape = (720, 1280)):

        #  Find the position of the car from the center
        # It will show if the car is 'x' meters from the left or right

        position = image_shape[1]/2
        left  = np.min(pts[(pts[:,1] < position) & (pts[:,0] > 700)][:,1])
        right = np.max(pts[(pts[:,1] > position) & (pts[:,0] > 700)][:,1])
        center = (left + right)/2

        # Define conversions in x and y from pixels space to meters,
        # assume the lane is about 30 meters long and 3.7 meters wide
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        return (position - center)*xm_per_pix

    @staticmethod
    def find_nearest(array, value):
        # Function to find the nearest point from array
        if len(array) > 0:
            idx = (np.abs(array - value)).argmin()
            return array[idx]

    @staticmethod
    def find_peaks(image, y_window_top, y_window_bottom, x_left, x_right):
        # Find the histogram from the image inside the window
        histogram = np.sum(image[y_window_top:y_window_bottom, :], axis=0)
        # Find the max from the histogram
        if len(histogram[int(x_left):int(x_right)]) > 0:
            return np.argmax(histogram[int(x_left):int(x_right)]) + x_left
        else:
            return (x_left + x_right) / 2

    @staticmethod
    def sanity_check(lane, curverad, fitx, fit):
        # Sanity check for the lane
        lane.current_fit = fit

        if abs(curverad - 2000) / 2000 < 2:
            lane.detected = True

            #Keep a running average over 3 frames
            if len(lane.recent_xfitted) > 5 and lane.recent_xfitted:
                lane.recent_xfitted.pop()
                lane.recent_fits.pop()

            lane.recent_xfitted.append(fitx.reshape(1,-1))
            lane.recent_fits.append(fit.reshape(1,-1))

            if len(lane.recent_xfitted) > 1:
                lane.bestx = np.mean(np.vstack(lane.recent_xfitted),axis=1)
                lane.best_fit = np.mean(np.vstack(lane.recent_fits),axis=1)

            lane.bestx = fitx
            lane.best_fit = fit

            return lane.bestx

        else:
            lane.detected=False

        return  lane.bestx if lane.bestx is not None else lane.current_fit

    # Takes care of fitting right and left lanes to the image.
    def _fit_lanes(self, image):

        left_lane, right_lane = self.left_lane, self.right_lane

        if left_lane.detected and right_lane.detected:
            left_fit, right_fit, yvals, left_fitx, right_fitx = \
                search_around_previous_fit(image, left_lane.best_fit, right_lane.best_fit)
        else:
            left_fit, right_fit, yvals, left_fitx, right_fitx = find_window_centroids(image, nwindows=8)

        # Find curvatures
        left_curverad = self.get_curvature(yvals, left_fitx)
        right_curverad = self.get_curvature(yvals, right_fitx)

        left_fitx = self.sanity_check(left_lane, left_curverad, left_fitx, left_fit)
        right_fitx = self.sanity_check(right_lane, right_curverad, right_fitx, right_fit)

        return yvals, left_fitx, right_fitx, left_curverad

    @staticmethod
    def draw_poly(image, warped, yvals, left_fitx, right_fitx, Minv, curvature):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        # Put text on an image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Radius of Curvature: {} m".format(int(curvature))
        cv2.putText(result,text,(400,100), font, 1,(255,255,255),2)
        # Find the position of the car
        pts = np.argwhere(newwarp[:,:,1])
        position = LaneFinder.get_position(pts)
        if position < 0:
            text = "Vehicle is {:.2f} m left of center".format(-position)
        else:
            text = "Vehicle is {:.2f} m right of center".format(position)
        cv2.putText(result,text,(400,150), font, 1,(255,255,255),2)
        return result

    def process_image(self, image):
        """
        Main function of the class which ingests the next image
        :param image: 
        :return: 
        """

        # 1. remove distortion
        undist = undistort(image)
        # 2. Apply segmentation to the image to create black and white image
        img = segment(undist)
        # 3. Warp the image to make lanes parallel to each other
        top_down, perspective_M, perspective_Minv = corners_unwarp(img)
        # 4. Find the lines fitting to left and right lanes
        yvals, left_fitx, right_fitx,  curvature = self._fit_lanes(top_down)
        # 5. Return the original image with colored region
        return self.draw_poly(image, top_down, yvals, left_fitx, right_fitx, perspective_Minv, curvature)


if __name__ == "__main__":
    ## Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML

    # Set up lines for left and right
    LF = LaneFinder()
    white_output = 'output2.mp4'
    clip1 = VideoFileClip("project_video.mp4")

    white_clip = clip1.fl_image(LF.process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
