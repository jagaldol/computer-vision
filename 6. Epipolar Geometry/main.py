import numpy as np
import matplotlib.pyplot as plt

img1 = plt.imread('./data/graffiti_a.jpg')
img2 = plt.imread('./data/graffiti_b.jpg')

cor1 = np.load("./data/graffiti_a.npy")
cor2 = np.load("./data/graffiti_b.npy")

def compute_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)
        
    F = None
    ### YOUR CODE BEGINS HERE
    
    # A container
    A = np.array([]).reshape(0, 9)

    for i in range(len(x1[0])):
        # for each matching points
        x = np.array([[x1[0][i], x1[1][i], x1[2][i]]])
        x_p = np.array([[x2[0][i]], [x2[1][i]], [x2[2][i]]])
        row = x_p @ x
        row = np.array([row.flatten()])
        # add row to A
        A = np.vstack([A, row])
    
    # calculate F with eigen values
    eig_val, eig_vec = np.linalg.eig(A.T @ A)
    min_index = eig_val.argmin()
    F = np.array(eig_vec[:, min_index]).reshape((3, 3))
    # constrain F: make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    
    S_p = np.diag([S[0], S[1], 0])

    F = U @ S_p @ V
    ### YOUR CODE ENDS HERE
    
    return F


def compute_norm_fundamental(x1,x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        exit(1)
    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = T1 @ x1
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = T2 @ x2

    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = T2.T @ F @ T1
    
    return F


def compute_epipoles(F):
    e1 = None
    e2 = None
    ### YOUR CODE BEGINS HERE
    U, S, V = np.linalg.svd(F) # solve Fe1 = 0 by SVD
    e1 = V[-1]
    e1 = e1/e1[2] # normalization
    
    U, S, V = np.linalg.svd(F.T) # solve F_te2 = 0 by SVD
    e2 = V[-1]
    e2 = e2/e2[2] # normalizaiton
    ### YOUR CODE ENDS HERE
    
    return e1, e2


def draw_epipolar_lines(img1, img2, cor1, cor2):
    F = compute_norm_fundamental(cor1, cor2)

    e1, e2 = compute_epipoles(F)
    ### YOUR CODE BEGINS HERE
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1)
    axes[0].title.set_text('Image 1')
    axes[0].set_xlim([0, img1.shape[1]])
    axes[0].set_ylim([img1.shape[0], 0])
    
    axes[1].imshow(img2)
    axes[1].title.set_text('Image 2')
    axes[1].set_xlim([0, img2.shape[1]])
    axes[1].set_ylim([img2.shape[0], 0])
    for i in range(len(cor1[0])):
        colors = np.random.random(3)
        # y = ax + b
        x1, y1 = cor1[0][i], cor1[1][i]
        a1 = (y1 - e1[1]) / (x1 - e1[0])
        b1 = y1 - a1 * x1

        x2, y2 = cor2[0][i], cor2[1][i]
        a2 = (y2 - e2[1]) / (x2 - e2[0])
        b2 = y2 - a2 * x2

        # 앞 끝점에 대해 계산
        min_x1 = 0
        min_y1 = a1 * min_x1 + b1
        max_x1 = img1.shape[1]
        max_y1 = a1 * max_x1 + b1
        min_x2 = 0
        min_y2 = a2 * min_x2 + b2
        max_x2 = img2.shape[1]
        max_y2 = a2 * max_x2 + b2

        # scatter
        axes[0].scatter(x1, y1, color=colors, s=100, marker='.')
        axes[1].scatter(x2, y2, color=colors, s=100, marker='.')
        
        # draw line
        axes[0].plot([min_x1, max_x1], [min_y1, max_y1], color=colors, linewidth=2)
        axes[1].plot([min_x2, max_x2], [min_y2, max_y2], color=colors, linewidth=2)

    plt.show()
    ### YOUR CODE ENDS HERE

    return

draw_epipolar_lines(img1, img2, cor1, cor2)
