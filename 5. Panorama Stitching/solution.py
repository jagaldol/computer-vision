import numpy as np
import math
import random


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
    inliers_num_best = 0
    h = None
    for _ in range(num_iter):
        # Sample 4 matches
        # rand_4 = random.sample(range(len(xy_src)), k=4)
        rand_4 = [random.randint(0,xy_src.shape[0]-1) for _ in range(4)]
        A = []
        for rand in rand_4:
            x, y = xy_src[rand]
            
            x_p, y_p = xy_ref[rand]            
            A.append([x, y, 1, 0, 0, 0, -x_p*x, -x_p*y, -x_p])
            A.append([0, 0, 0, x, y, 1, -y_p*x, -y_p*y, -y_p])

        A = np.array(A)
        # Calculate Homography Matrix
    
        eig_val, eig_vec = np.linalg.eig(A.T.dot(A))
        min_index = eig_val.argmin()
        homography = np.array(eig_vec[:, min_index]).reshape((3, 3))
        # Move src points into ref space
        xy_out = KeypointProjection(xy_src, homography)
        # Calculate distance
        dists = np.sqrt(np.sum((xy_out - xy_ref)**2, axis=1))
        # Calculate inliers Num
        inliers_num = np.count_nonzero(dists <= tol)
        if inliers_num > inliers_num_best:
            h = homography
            inliers_num_best = inliers_num
    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h