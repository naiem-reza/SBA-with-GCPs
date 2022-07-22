"""
# Sparse-Bundle-Adjustments-with-GCPs


This python implementation of Sparse bundle adjustment based on the sparse Levenberg-Marquardt algorithm with Ground Control Points (GCPs). 
A bundle adjusmtent problem arises in 3-D reconstruction and it can be formulated as follows (taken from https://en.wikipedia.org/wiki/Bundle_adjustment):

To derive accurate 3D geospatial information from imagery, it is necessary to establish the camera's interior and exterior orientation parameters. 
Interior orientation parameters (IOPs), which contain the internal sensor elements such as principal distance, principal point coordinates, and lens distortions, are specified through a camera calibration procedure. 
EOPs, which define the position and orientation of the camera at the point of exposure in a mapping frame, can be established using either Ground Control Points (GCPs) through a Aerial triangulation (AT) process. 
AT is one of the most critical steps in aerial photogrammetry to estimate the Tie points Object coordinates (OC), EOPs, and  IOP which is performed with the bundle Adjustment (BA).

The basic photogrammetric principal geometry consists of three geometric entities: object space points (3D points), 
corresponding image points (2D points), and perspective centers. Such a geometry can be formulized with collinearity equations. 

 R_{ij} (i = 1,2,3,... , j = 1,2,3,...) stands for the nine elements of the rotation matrix , which can be modeled by three rotation Euler angles omega,
 phi and kappa. 
 
R = r_{11} & r_{12} & r_{13}
    r_{21} & r_{22} & r_{23}
    r_{31} & r_{32} & r_{33}

    [P]        [X - X0]
    [S]  = R . [Y - Y0]
    [Q]        [Z - Z0]

 
X_0,Y_0 and Z_0 are the translation parameters for the camera station. Meanwhile, X,Y and Z are are the object point coordinates usually given in meters in the mapping reference system (i.e., Earth-fixed coordinate system, in the case UTM)

The additional parameters related to lens distortions, coordinates of the principal point (i.e., the point closest to the projection center), 
and sensor distortions can, in practice, be used to recover the theoretical collinearity condition between image points, camera position, and object point.

To estimate the 6 EOPs, the image coordinates are first rectifed using the calculated IOPs of the digital camera, consisting of the principal point (c_x,c_x) (in pixel), the focal length f (in pixel), 
the coefcients of radial distortion (K_1, K_2,K_3) (in pixel) and the coefcients of decentring distortion (P1,P2) (pixel) and affinity and non-orthogonality (skew) coefficients (B1, B2) (in pixels).  
According to equations, the image measurements (u, v) (pixel) are rectifed to (u', v') (pixel) according to

 
x = - P / Q

y = S/ Q

r= sqrt{x^2+y^2}

x' = x(K_1r^2 + K_2r^4 + K_3r^6) + (P_1(r^2+2x^2) + 2P_2xy)

y' = y(K_1r^2 + K_2r^4 + K_3r^6) + (P_2(r^2+2y^2) + 2P_1xy)

u' = 0.5w + c_x + x'f + x'B1 + y'B2

v' = 0.5h + c_y + y'f

f_{x} = u - u'

f_{y} = v - v'

"""




import os
import copy
import time
import numpy as np
import Lib
import matplotlib.pyplot as plt
from ismember import ismember
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

# --------------------------------------------------------------------- User inputs
data_folder_name = 'data'
lnda = 10
convergence_criteria = 1     # pixel
Max_iter = 1000

# --------------------------------------------------------------------- Import Data
# initial value of EOPs, IOPs, 3D coordinate of Ties, image Observation
camera_params, IOP, initial_tie, Control, img_Observation, obj_Observation = Lib.Import_Data(os.getcwd() + '\\' + data_folder_name)

# --------------------------------------------------------------------- Find Image Observation of Gcp ans Tie point
points_3d = np.vstack((initial_tie[:, 1:], Control[:, 1:]))
points_2d = img_Observation[:, 2:]
camera_indices = img_Observation[:, 0].astype(int)
Tie_indices = img_Observation[:, 1].astype(int)
[I1, I2] = ismember(img_Observation[:,1], Control[:,0])
GCPs_indices = img_Observation[I1, 1].astype(int)

n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]
n = 6 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]
print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print('')
print("Total number of parameters: {}".format(n))
print("Total number of Observation: {}".format(m))
print('')

# 7 --------------------------------------------------------------------- Approximate Value
x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

print('   iter       lambda      SSE(pix)       RMSE(pix)       Mean(pix)     Time(min)      Sum Time(h)')
print('----------------------------------------------------------------------------------------------------')
tt = time.time()
rmse = []
for j in range(Max_iter):
    t0 = time.time()

    # --------------------------------------------------------------------- jacobin matrix and Loss vector
    A = Lib.JAC_1(x0, n_cameras, n_points, camera_indices, Tie_indices, GCPs_indices, IOP)
    dl, reproject = Lib.FUNC_1(x0, n_cameras, n_points, camera_indices, Tie_indices, GCPs_indices, points_2d, obj_Observation, IOP)

    C = A[:, :n_cameras * 6]
    P = A[:, n_cameras * 6:]
    r = -(A.T @ dl)
    rc = r[:n_cameras * 6]
    rp = r[n_cameras * 6:]
    rmse.append(np.sqrt(np.mean(dl**2)))

    # ---------------------------------------------------------------------  Damp Parameter
    diag = (A.T @ A).diagonal()
    D = lil_matrix((len(diag), len(diag)), dtype=float)
    D[np.arange(len(diag)), np.arange(len(diag))] = diag
    Dc = D[:n_cameras * 6, :n_cameras * 6]
    Dp = D[n_cameras * 6:, n_cameras * 6:]

    # --------------------------------------------------------------------- Normal Matrix , levenberg-marquardt
    if lnda <= 1e-10:
        ld = 0
    else:
        ld = copy.deepcopy(lnda)
    U = (C.T @ C) + ld*Dc
    W = C.T @ P
    Wt = P.T @ C
    V = (P.T @ P) + ld*Dp
    iV = Lib.invers_block_diag_sparse(V)

    # --------------------------------------------------------------------- RNE
    AA = U - (W @ iV @ Wt)
    ll = rc - (W @ iV @ rp)
    dc = lsqr(AA, ll)
    dp = iV @ (rp - Wt @ dc[0])
    x0 += np.hstack((dc[0], dp))

    # --------------------------------------------------------------------- Print information of iterations
    if j == 0 or (j+1)%10 == 0 or np.sqrt(np.mean(dl**2)) < convergence_criteria:
        h = ((time.time()-tt)//3600)
        min = np.round(((time.time()-tt)/3600 - (time.time()-tt)//3600)*60)
        if lnda <= 1e-10:
            ld = 0
        else:
            ld = copy.deepcopy(lnda)

        str1 = '%6g       %6g       %5.2e' % (j + 1, ld, 0.5 * (sum(dl ** 2)))
        str2 = '       %5.3e       %5.2f       %5.2f       %5g:%2g' % (np.sqrt(np.mean(dl**2)), np.mean(np.abs(dl)), ((time.time() - t0) / 60), h, min)
        print(str1, str2)

    # --------------------------------------------------------------------- Updating levenberg-marquardt
    dl_new, reproject = Lib.FUNC_1(x0, n_cameras, n_points, camera_indices, Tie_indices, GCPs_indices, points_2d, obj_Observation, IOP)
    if np.sqrt(np.mean(dl_new**2)) <= np.sqrt(np.mean(dl**2)):
        lnda /= 10
    else:
        lnda *= 10

    if np.sqrt(np.mean(dl**2)) < convergence_criteria:
        break


Final_camera_params = x0[:n_cameras * 6].reshape((n_cameras, 6))
Final_points_3d = x0[n_cameras * 6:].reshape((n_points, 3))

fig = plt.figure()
plt.plot(np.abs(np.asarray(rmse)), 'r')
plt.title('RMSE of Residuals')
plt.xlabel('Iteration')
plt.ylabel('image residuals (mm)')
plt.grid(True)
