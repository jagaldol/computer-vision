import numpy as np
import cv2
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    largest_set = []
    orient_agreement = orient_agreement / 180 * math.pi
    for i in range(10) : # repeat 10 times
        rand = random.randrange(0, len(matched_pairs)) # generate random number
        choice = matched_pairs[rand]
        orientation = (keypoints1[choice[0]][3] - keypoints2[choice[1]][3]) % (2 * math.pi) # calculation first-orientation
        scale = keypoints2[choice[1]][2] / keypoints1[choice[0]][2] # clacualation first-scale ratio
        temp = []
        for j in range(len(matched_pairs)): # calculate the number of all cases
            if j is not rand:
                # calculation second-orientation
                orientation_temp = (keypoints1[matched_pairs[j][0]][3] - keypoints2[matched_pairs[j][1]][3]) % (2 * math.pi)
                # calculation second-scale ratio
                scale_temp = keypoints2[matched_pairs[j][1]][2] / keypoints1[matched_pairs[j][0]][2]
                # check degree error
                if (abs(orientation - orientation_temp) < orient_agreement) or (abs(orientation - orientation_temp) > (2 * math.pi) - orient_agreement):
                    # check sacle error
                    if((scale - scale * scale_agreement < scale_temp) and (scale_temp < scale + scale * scale_agreement)):
                        temp.append([i, j])
        if len(temp) > len(largest_set): # choice best match
            largest_set = temp
    for i in range(len(largest_set)):
        largest_set[i] = (matched_pairs[largest_set[i][1]][0], matched_pairs[largest_set[i][1]][1])

    ## END
    assert isinstance(largest_set, list)
    return largest_set



def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    ## the following is just a placeholder to show you the output format
    y1 = descriptors1.shape[0] # save descriptors1 size
    y2 = descriptors2.shape[0] # save descriptors2 size
    temp = np.zeros(y2) # make an array of descriptors2 size
    matched_pairs = []

    for i in range(y1):
        for j in range(y2):
            temp[j] = math.acos(np.dot(descriptors1[i], descriptors2[j])) # calculate the number of all cases
        compare = sorted(range(len(temp)), key = lambda k : temp[k]) # comparison to find the best
        if (temp[compare[0]] / temp[compare[1]]) < threshold: # check the best match
            matched_pairs.append([i, compare[0]]) # best match | i = descriptors1 | j = descriptors2
    ## END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    # Make homogeneous
    xy_points_hg = np.append(xy_points, np.ones((xy_points.shape[0], 1)), axis=1)
    # Apply h matrix
    xy_points_proj = h.dot(xy_points_hg.T).T
    # Avoid Divide By Zero
    is_zero_points = xy_points_proj[:, 2] == 0
    xy_points_proj[is_zero_points, 2] = 1e-10

    # Return to Regular Coordinate
    xy_points_out = xy_points_proj[:, :2] / xy_points_proj[:, 2:]

    # END
    return xy_points_out

def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0

    # START



    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
