import os
import cv2
import json
import copy
import numpy as np
from plantcv import plantcv as pcv
from scipy.interpolate import interp1d

def dilate_image(image, kernel_size, iterations=1):
    # Create a structuring element (kernel) for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform dilation
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image

def horizontal_dilation(image, kernel_width=5,iterations=1):
    # Create a horizontal kernel for dilation
    kernel = np.ones((1, kernel_width), np.uint8)
    # Perform dilation
    dilated_image = cv2.dilate(image, kernel, iterations)
    return dilated_image

def vertical_dilation(image, kernel_width=5,iterations=1):
    # Create a vertical kernel for dilation
    kernel = np.ones((kernel_width, 2), np.uint8)
    # Perform dilation
    dilated_image = cv2.dilate(image, kernel, iterations)
    return dilated_image

# New Utils
def deformat(listofpoints):
    # Input : [[[x1,y1],[[x2,y2]],[[x3,y3]]....]
    # Output : [ [x1,y1], [x2,y2],[x3,y3]....]
    output = [ pt[0].tolist() for pt in listofpoints ]
    return output

# Supply the raw image here
def cleanImageFindContours(patch,threshold):
  try:
    patch = np.uint8(patch)
    patch = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(patch,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)<1:
      print('No contours in the raw image!')
      return patch
    # Else sort them
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    areaList = [cv2.contourArea(c) for c in cntsSorted]
    maxArea = max(areaList)
    sortedContours = [deformat(c) for c in cntsSorted if cv2.contourArea(c)>np.int32(threshold*maxArea)]
    return sortedContours

  except Exception as exp :
    print('Error in figuring out the clean contours : {} '.format(exp))
    return None


def uniformly_sampled_line(points):
    num_points = min(len(points),250)
    # Separate x and y coordinates from the given points
    x_coords, y_coords = zip(*points)

    # Calculate the cumulative distance along the original line
    distances = np.cumsum(np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2))
    distances = np.insert(distances, 0, 0)  # Add the initial point (0, 0) distance

    # Create a linear interpolation function for x and y coordinates
    interpolate_x = interp1d(distances, x_coords, kind='linear')
    interpolate_y = interp1d(distances, y_coords, kind='linear')

    # Calculate new uniformly spaced distances
    new_distances = np.linspace(0, distances[-1], num_points)

    # Interpolate new x and y coordinates using the uniformly spaced distances
    new_x_coords = interpolate_x(new_distances)
    new_y_coords = interpolate_y(new_distances)

    # Create a list of new points
    new_points = [[np.int32(new_x_coords[i]), np.int32(new_y_coords[i])] for i in range(num_points)]

    return new_points

def find_corner_points(points):
    if not points:
        return None, None
    # Sort the points based on their x-coordinate
    sorted_points = sorted(points, key=lambda point: point[0])
    leftmost_point = list(sorted_points[0])
    rightmost_point = list(sorted_points[-1])
    return leftmost_point, rightmost_point

def generateScribble(H,W,polygon):
    # Generate Canvas
    canvas = np.zeros((H,W))
    # Mark the polygon on the canvas
    leftmost_point, rightmost_point = find_corner_points(polygon)
    poly_arr = np.asarray(polygon,dtype=np.int32).reshape((-1,1,2))
    canvas = cv2.fillPoly(canvas,[poly_arr],(255,255,255))
    # Scribble generation
    skeleton = pcv.morphology.skeletonize(canvas)
    pruned_skeleton,_,segment_objects = pcv.morphology.prune(skel_img=skeleton,size=100)
    scribble = np.asarray(segment_objects[0],dtype=np.int32).reshape((-1,2))
    scribble=scribble.tolist()
    scribble = uniformly_sampled_line(scribble)
    if leftmost_point is not None and rightmost_point is not None :
      scribble.append(leftmost_point)
      scribble.append(rightmost_point)
      scribble = sorted(scribble, key=lambda point: point[0])
    return scribble


def polygon_to_distance_mask(pmask,threshold=60):
    # Read the polygon mask image as a binary image
    polygon_mask = cv2.cvtColor(pmask,cv2.COLOR_BGR2GRAY)

    # Ensure that the mask is binary (0 or 255 values)
    _, polygon_mask = cv2.threshold(polygon_mask, 128, 255, cv2.THRESH_BINARY)

    # Compute the distance transform
    distance_mask = cv2.distanceTransform(polygon_mask, cv2.DIST_L2, cv2.DIST_MASK_5)

    # Normalize the distance values to 0-255 range
    distance_mask = cv2.normalize(distance_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Threshold the image
    src = copy.deepcopy(distance_mask)
    src[src<threshold]=0
    src[src>=threshold]=255
    src = np.uint8(src)
    return src


def average_coordinates(hull):
    # Calculate the average x and y coordinates of all points in the hull
    num_points = len(hull)
    avg_x = sum(pt[0][0] for pt in hull) / num_points
    avg_y = sum(pt[0][1] for pt in hull) / num_points
    return avg_x, avg_y

def combine_hulls_on_same_level(contours, threshold=45):
    combined_hulls = []
    hulls = [cv2.convexHull(np.array(contour)) for contour in contours]

    # Sort the hulls by the average y-coordinate of all points
    sorted_hulls = sorted(hulls, key=lambda hull: average_coordinates(hull)[1])

    current_combined_hull = sorted_hulls[0]
    for hull in sorted_hulls[1:]:
        # Check if the current hull is on the same horizontal level as the combined hull
        if abs(average_coordinates(hull)[1] - average_coordinates(current_combined_hull)[1]) < threshold:
            # Merge the hulls by extending the current_combined_hull with hull
            current_combined_hull = np.vstack((current_combined_hull, hull))
        else:
            # Hull is on a different level, add the current combined hull to the result
            combined_hulls.append(current_combined_hull)
            current_combined_hull = hull

    # Add the last combined hull
    combined_hulls.append(current_combined_hull)
    nethulls = [cv2.convexHull(np.array(contour)) for contour in combined_hulls]
    return combined_hulls,nethulls

def find_text_bounding_box(image, factor=0.4):
    # Find contours in the image
    image = image[:,:,0].astype('uint8')
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box enclosing all contours
    if contours:
        all_points = [point for contour in contours for point in contour]
        min_rect = cv2.minAreaRect(np.array(all_points))

        # Draw the minimum bounding rectangle on the image
        image_with_box = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rect = list(min_rect)
        print('width, height? :', rect[1])
        print('angle :', rect[2])
        max_idx = 0
        if rect[2] > 80:
            max_idx = 1
        rect[1] = list(rect[1])
        rect[1][max_idx] -= int(factor*rect[1][max_idx])
        rect[1] = tuple(rect[1])
        print('updated width, height? :', rect[1])
        min_rect = tuple(rect)
        box = cv2.boxPoints(min_rect)
        box = np.int0(box)
        image_with_box = cv2.drawContours(image_with_box, [box], 0, (0, 255, 0), 2)

        return image_with_box,box, int(factor*rect[1][max_idx])

def get_box(img, cut):
    img_ = vertical_dilation(img,kernel_width=80,iterations=10)
    big_contour = cleanImageFindContours(img_,threshold = 0.50)
    big_contour = np.array(big_contour[0], dtype=np.int32)
    rect = cv2.minAreaRect(big_contour)
    rect = list(rect)
    max_idx = rect[1].index(max(rect[1]))
    rect[1] = list(rect[1])
    rect[1][max_idx] -= cut
    rect[1] = tuple(rect[1])
    rect = tuple(rect)
    box = cv2.boxPoints(rect)
    cv2.drawContours(img,np.array([box], np.int32),0,(0,0,255),3)
    return img, box


def get_subset(box, img):
    source_image = np.array(img, dtype=np.int32)
    mask = np.zeros(img.shape[:-1], dtype=np.uint8)
    cv2.fillPoly(mask, np.array([box], dtype=np.int32), 255)

    polygon_part = cv2.bitwise_and(source_image, source_image, mask=mask)

    canvas = np.zeros(source_image.shape, dtype=np.int32)

    canvas = cv2.bitwise_or(canvas, polygon_part)
    return canvas

def find_leftmost_rightmost_points(points):
    points = format(points)
    leftmost = min(points, key=lambda p: p[0])
    rightmost = max(points, key=lambda p: p[0])
    return leftmost, rightmost

def format(List):
    if len(np.array(List).shape)!= 2:
        List = List.reshape(List.shape[0], List.shape[-1])
    return List
