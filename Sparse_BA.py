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
data_folder_name = 'data3'
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