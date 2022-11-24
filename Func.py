import os
import copy
import time
import math
import scipy
import numpy as np
import pandas as pd
import xmltodict
from scipy.stats import norm
from ismember import ismember
import matplotlib.pyplot as plt
from matplotlib import image
from numpy.linalg import inv as npinv
from scipy.sparse import lil_matrix, bsr_matrix
from scipy.sparse.linalg import spsolve

# --------------------------------- Read Cameras EOPs and IOP File
def ReadEOPFile(eop_path):
    with open(eop_path, 'r') as ALL_EOP:
        EOPFile = ALL_EOP.readlines()
    neop = len(EOPFile) - 2
    PhotoID = [];    X = [];    Y = [];    Z = [];    omega = [];    phi = [];    kappa = [];    r11 = []
    r12 = [];    r13 = [];    r21 = [];    r22 = [];    r23 = [];    r31 = [];    r32 = [];    r33 = []
    for row in EOPFile[2: len(EOPFile)]:
        PhotoID.append(str(row.split('\t')[0]))
        X.append(float(row.split('\t')[1]))
        Y.append(float(row.split('\t')[2]))
        Z.append(float(row.split('\t')[3]))
        omega.append(float(row.split('\t')[4]))
        phi.append(float(row.split('\t')[5]))
        kappa.append(float(row.split('\t')[6]))

    #  imgeop = [XO, YO, ZO, Omega, Phi, Kappa, r11, r12, ... , r32, r33]
    imgeop = np.vstack((np.array(X),np.array(Y),np.array(Z),np.array(omega),np.array(phi),np.array(kappa))).transpose()
    return imgeop
def ReadIOPFile(cam_path):
    xml_data = open(cam_path, 'r').read()  # Read data
    data = xmltodict.parse(xml_data)  # Parse XML
    iop = []
    for i in data['calibration']:
        child = data['calibration'][i]
        iop.append(child)

    if len(iop) <= 5:
        w = float(iop[1])
        h = float(iop[2])
        f = float(iop[3])
        IOP = [w, h, f, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    else:
        w = float(iop[1])
        h = float(iop[2])
        f = float(iop[3])
        cx = float(iop[4])
        cy = float(iop[5])
        b1 = float(iop[6])
        b2 = float(iop[7])
        k1 = float(iop[8])
        k2 = float(iop[9])
        k3 = float(iop[10])
        p1 = float(iop[11])
        p2 = float(iop[12])
        IOP = [w, h, f, cx, cy, b1, b2, k1, k2, k3, p1, p2]
    return IOP
def ReadIOPFile_Australis(cam_path):
    with open(cam_path, 'r') as ALL_IOP:
        IOPFile = ALL_IOP.readlines()
    w = float(IOPFile[15][7:11])
    h = float(IOPFile[14][7:11])
    f = float(IOPFile[19][13:19])
    cx = float(IOPFile[20][12:19])
    cy = float(IOPFile[21][12:19])
    k1 = float(IOPFile[22][7:19])
    k2 = float(IOPFile[23][7:19])
    k3 = float(IOPFile[24][7:19])
    k4 = 0
    p1 = float(IOPFile[25][7:19])
    p2 = float(IOPFile[26][7:19])
    b1 = float(float(IOPFile[27][7:19]))
    b2 = float(float(IOPFile[28][7:19]))
    ps = float(float(IOPFile[14][22:32]))

    IOP = [h, w, f, cx, cy, b1, b2, k1, k2, k3, k4, p1, p2]
    return IOP
def ReadEOPFile1(eop_path):
    with open(eop_path, 'r') as ALL_EOP:
        EOPFile = ALL_EOP.readlines()
    neop = len(EOPFile) - 3
    PhotoID = []
    X = []
    Y = []
    Z = []
    omega = []
    phi = []
    kappa = []
    for row in EOPFile[2: len(EOPFile)]:
        PhotoID.append(str(row.split(',')[0]))
        X.append(float(row.split(',')[1]))
        Y.append(float(row.split(',')[2]))
        Z.append(float(row.split(',')[3]))
        omega.append(float(row.split(',')[4]))
        phi.append(float(row.split(',')[5]))
        kappa.append(float(row.split(',')[6]))

    #  imgeop = [XO, YO, ZO, Omega, Phi, Kappa, r11, r12, ... , r32, r33]
    imgeop = np.vstack((np.array(X), np.array(Y), np.array(Z), np.array(omega), np.array(phi), np.array(kappa))).transpose()
    return imgeop

# --------------------------------- XYZ of Tie points
def ReadXYZTieFile(xyz_path):
    ID = [];    xx = [];    yy = [];    zz = []
    with open(xyz_path, 'r') as ALL_Tie:
        TieFile = ALL_Tie.readlines()
    for row in TieFile[0: len(TieFile)]:
        ID.append(int(row.split(',')[0]))
        xx.append(float(row.split(',')[1]))
        yy.append(float(row.split(',')[2]))
        zz.append(float(row.split(',')[3]))
    #  XYZtie = [Number , X , Y , Z]
    XYZTie = np.vstack((np.array(ID), np.array(xx), np.array(yy), np.array(zz))).transpose()
    return XYZTie

# --------------------------------- Read image observation of Tie points
def readimgTieobs(obs_path,TieXYZ):
    os.chdir(obs_path)
    Tie_obs = []

    # iterate through all file
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{obs_path}/{file}"
            ID = []
            xx = []
            yy = []
            with open(file_path, 'r') as f:
                cor = f.readlines()
            for row in cor[0: len(cor)]:
                ID.append(int(row.split(',')[0]))
                xx.append(float(row.split(',')[1]))
                yy.append(float(row.split(',')[2]))
            xyimg = np.vstack((np.array(ID), np.array(xx), np.array(yy))).transpose()
            [Ixyz , Ixy] = ismember(TieXYZ[:,0] , xyimg[:,0])
            xyimg = xyimg[Ixy]
            Tie_obs.append(xyimg)
    return(Tie_obs)

# --------------------------------- Import Data
def Import_Data(dir_files):
    eop_path = dir_files + '/Step_1/eop_init.txt'
    eopref_path = dir_files + '/Step_1/eop_ref.txt'
    cam_path = dir_files + '/Step_1/iop_ref.xml'
    xyz_path = dir_files + '/Step_2/xyz.txt'
    Tieobs_path = dir_files + '/Step_2/imgdata'

    EOP_ref = ReadEOPFile(eopref_path)
    EOP_init = ReadEOPFile1(eop_path)
    IOP = ReadIOPFile(cam_path)
    TieXYZ = ReadXYZTieFile(xyz_path)
    TieObs = readimgTieobs(Tieobs_path,TieXYZ)
    EOP = EOP_init + (EOP_ref - EOP_init) / 2

    return EOP_ref, EOP, IOP, TieXYZ, TieObs

def Control_from_xml(dir_files, Control_start_id):
    gcp_path = dir_files + '/Step_1/gcp.xml'
    xml_data = open(gcp_path, 'r').read()  # Read data
    data = xmltodict.parse(xml_data)  # Parse XML
    marker = data['document']['chunk']['markers']['marker']
    obs = data['document']['chunk']['frames']['frame']['markers']['marker']
    for i in range(len(marker)):
        marker[i]['@id'] = str(i)
        marker[i]['@label'] = str(i)
        obs[i]['@marker_id'] = str(i)

    CP_LB, CH_LB, Control, Check, Control_Obs, Check_Obs = [], [], [], [], [], []
    for i in range(len(marker)):
        CP_LB.append(marker[i]['@label'])
        Control.append(np.asarray([marker[i]['@id'], marker[i]['reference']['@x'], marker[i]['reference']['@y'], marker[i]['reference']['@z']], dtype=float))
        for j in range(len(obs[i]['location'])):
            Control_Obs.append(np.asarray([obs[i]['location'][j]['@camera_id'], marker[i]['@id'], obs[i]['location'][j]['@x'], obs[i]['location'][j]['@y']], dtype=float))

    Control = np.asarray(Control)
    Cnt = pd.DataFrame(np.asarray(Control_Obs)).sort_values([1, 0])
    unique_index = pd.unique(Cnt[1])
    Cnt[1] = np.digitize(Cnt[1], unique_index) - 1 + Control_start_id
    Control[:, 0] = np.digitize(Control[:, 0], unique_index) - 1 + Control_start_id
    Control_Obs = Cnt.to_numpy()

    return Control, Control_Obs

def preprocess(TieXYZ, TieObs, EOP):
    PI = math.pi
    UV_ud = []
    imgnum = len(EOP)
    for i in range(imgnum):
        num = len(TieObs[i])
        if len(UV_ud) == 0:
            UV_ud = np.column_stack(((i * np.ones([num])).astype(int), TieObs[i]))
        else:
            UV_ud = np.vstack((UV_ud, np.column_stack(((i * np.ones([num])).astype(int), TieObs[i]))))

    Observation_Tie = pd.DataFrame(UV_ud).sort_values([1, 0])
    unique_index = pd.unique(Observation_Tie[1])
    Observation_Tie[1] = np.digitize(Observation_Tie[1], unique_index) - 1
    TieXYZ[:, 0] = np.digitize(TieXYZ[:, 0], unique_index) - 1
    camera_params = np.column_stack((EOP[:, :3], EOP[:, 3:]*(PI/180)))
    return Observation_Tie.to_numpy(), TieXYZ, camera_params

def PLOT_iter(RES_IMG, PHI, RES_OBJ):
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
    ax1.plot(np.abs(np.asarray(RES_IMG[:, 0])), 'r', label='x')
    ax1.plot(np.abs(np.asarray(RES_IMG[:, 1])), 'b', label='y')
    ax1.set_title('RMSE of Image Residuals')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('image residuals (pixel)')
    ax1.legend(loc="best")
    ax1.grid(True)

    ax2.plot(PHI, 'r')
    ax2.set_title('Phi')
    ax2.set_xlabel('Iteration')
    ax2.grid(True)

    ax3.plot(np.abs(np.asarray(RES_OBJ[:, 0])), 'r', label='X')
    ax3.plot(np.abs(np.asarray(RES_OBJ[:, 1])), 'g', label='Y')
    ax3.plot(np.abs(np.asarray(RES_OBJ[:, 2])), 'b', label='Z')
    ax3.set_title('RMSE of Object Residuals')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Object residuals (m)')
    ax3.legend(loc="best")
    ax3.grid(True)

def Plot_residual_hist(dl_last):
    fig, [ax1 , ax2] = plt.subplots(1,2)
    ax1.hist(dl_last, bins=100, density=True)
    ax1.grid(True)
    [mean_fit, std_fit] = scipy.stats.norm.fit(dl_last)
    x = np.linspace(dl_last.min(), dl_last.max())
    ax1.plot(x, scipy.stats.norm.pdf(x, mean_fit, std_fit))
    ax1.set_xlabel('residuals of Tie Distance')
    ax1.set_ylabel('Number of Tie')
    ax1.set_title('Mean_fit:%0.2f' % mean_fit + '   std_fit:%0.2f' % std_fit)

    ax2.hist(dl_last, bins=100, density=True)
    ax2.grid(True)
    [scale_fit, mean_fit, std_fit] = scipy.stats.t.fit(dl_last)
    x = np.linspace(dl_last.min(), dl_last.max())
    ax2.plot(x, scipy.stats.t.pdf(x, scale_fit, mean_fit, std_fit))
    ax2.set_xlabel('residuals of Tie Distance')
    ax2.set_ylabel('Number of Tie')
    ax2.set_title('Scale_fit:%0.2f' % scale_fit + '   Mean_fit:%0.2f' % mean_fit + '   std_fit:%0.2f' % std_fit)

def show_residual(dir_files, Observation, Udpoint2d_pix, num):
    gcp_path = dir_files + '/Step_1/gcp.xml'
    xml_data = open(gcp_path, 'r').read()  # Read data
    data = xmltodict.parse(xml_data)  # Parse XML
    camera = data['document']['chunk']['cameras']['camera']
    cam_label = []
    for i in range(len(camera)):
        cam_label.append(camera[i]['@label'])

    udp = copy.deepcopy(Observation)
    udp[:, 2:] = Udpoint2d_pix
    udp = udp[np.argsort(udp[:, 0])]
    obs = copy.deepcopy(Observation)
    obs = obs[np.argsort(obs[:, 0])]

    fig = plt.figure()
    imgdir = dir_files + '/Image/' + cam_label[num] + '.JPG'
    data = image.imread(imgdir)
    [I1, I2] = ismember(udp[:, 0].astype(int), np.array(num).astype(int))

    xn = udp[I1, 2]
    yn = udp[I1, 3]
    x = obs[I1, 2]
    y = obs[I1, 3]

    x1 = [xn, x]
    y1 = [yn, y]
    plt.plot(x1, y1, color="black", linewidth=2)

    plt.plot(xn, yn, '.r', label='re-project')
    plt.plot(x, y, '.b', label='Observation')
    plt.imshow(data)
    plt.show()
    plt.legend()

def inverse_block_diag_sparse(A):
    V = bsr_matrix(A)
    size = V.blocksize
    iV = copy.deepcopy(V)

    data = V.data
    idata = np.zeros((len(data), size[0], size[1]))
    for k in range(len(data)):
        idata[k, :, :] = npinv(data[k])

    iV.data = idata
    return iV.tocsr()

def Weight(Tie_indices, Gcp_indices, sigma_img, sigma_obj):
    m = (len(Tie_indices) * 2) + len(Gcp_indices) * 3
    Weight = lil_matrix((m, m), dtype=float)
    ii = np.arange(len(Tie_indices) * 2)
    Weight[ii, ii] = 1 / sigma_img ** 2
    ii = np.arange(len(Tie_indices) * 2, len(Tie_indices) * 2 + len(Gcp_indices) * 3)
    Weight[ii, ii] = 1 / sigma_obj ** 2
    return Weight.tocsr()

def ObjFun(params, n_cameras, n_tiepoints, camera_indices, Tie_indices, Gcp_indices, IOP, points_2d, GCP):
    camera = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_tiepoints, 3))
    conpoints_3d = points_3d[Gcp_indices]

    XYZ = points_3d[Tie_indices]
    eop = camera[camera_indices]

    rr = []
    for i, val in enumerate(eop):
        Omega = val[3]
        Phi = val[4]
        Kapa = val[5]
        R = np.asarray([[np.cos(Kapa) * np.cos(Phi), np.cos(Omega) * np.sin(Kapa) + np.cos(Kapa) * np.sin(Omega) * np.sin(Phi), np.sin(Kapa) * np.sin(Omega) - np.cos(Kapa) * np.cos(Omega) * np.sin(Phi)],
                        [-np.cos(Phi) * np.sin(Kapa), np.cos(Kapa) * np.cos(Omega) - np.sin(Kapa) * np.sin(Omega) * np.sin(Phi), np.cos(Kapa) * np.sin(Omega) + np.cos(Omega) * np.sin(Kapa) * np.sin(Phi)],
                        [np.sin(Phi), -np.cos(Phi) * np.sin(Omega), np.cos(Omega) * np.cos(Phi)]])
        rr.append(R)

    t = XYZ - eop[:, :3]
    XYZ_cam = []
    for j in range(len(eop)):
        XYZ_cam.append((rr[j] @ t[j, :].T).T)
    XYZ_camera = np.asarray(XYZ_cam)

    W = IOP[0]
    H = IOP[1]
    f = IOP[2]
    cx = IOP[3]
    cy = IOP[4]
    B1 = IOP[5]
    B2 = IOP[6]
    K1 = IOP[7]
    K2 = IOP[8]
    K3 = IOP[9]
    P1 = IOP[10]
    P2 = IOP[11]

    xp = -(XYZ_camera[:, 0] / XYZ_camera[:, 2])
    yp = (XYZ_camera[:, 1] / XYZ_camera[:, 2])

    r = np.sqrt(xp ** 2 + yp ** 2)
    dx = xp * (1 + K1 * (r ** 2) + K2 * (r ** 4) + K3 * (r ** 6)) + (P1 * ((r ** 2) + 2 * (xp ** 2))) + (2 * P2 * (xp * yp))
    dy = yp * (1 + K1 * (r ** 2) + K2 * (r ** 4) + K3 * (r ** 6)) + (P2 * ((r ** 2) + 2 * (yp ** 2))) + (2 * P1 * (xp * yp))

    u = W * 0.5 + cx + dx * f + dx * B1 + dy * B2
    v = H * 0.5 + cy + dy * f

    f1 = points_2d[:, 0] - u
    f2 = points_2d[:, 1] - v

    res = np.column_stack((f1, f2)).ravel()
    resgcp = (conpoints_3d-GCP).ravel()

    return np.hstack((res, resgcp)), np.column_stack((u, v))

def Jac(params, n_cameras, n_tiepoints, camera_indices, Tie_indices, Gcp_indices, IOP):
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_tiepoints, 3))

    m = len(Tie_indices) * 2 + len(Gcp_indices) * 3
    n = n_cameras * 6 + n_tiepoints * 3
    A = lil_matrix((m, n), dtype=float)
    eop = camera_params[camera_indices]
    XYZ = points_3d[Tie_indices]

    f = IOP[2]
    B1 = IOP[5]
    B2 = IOP[6]

    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]

    X0 = eop[:, 0]
    Y0 = eop[:, 1]
    Z0 = eop[:, 2]

    cw = np.cos(eop[:, 3])
    cp = np.cos(eop[:, 4])
    ck = np.cos(eop[:, 5])
    sw = np.sin(eop[:, 3])
    sp = np.sin(eop[:, 4])
    sk = np.sin(eop[:, 5])
    
    t1 = (cw * sk + ck * sp * sw)
    t2 = (ck * cw - sk * sp * sw)
    t3 = (ck * sw + cw * sk * sp)
    t4 = (sk * sw - ck * cw * sp)
    T1 = ((Y - Y0) * t2 + (Z - Z0) * t3 - cp * sk * (X - X0))
    T2 = (sp * (X - X0) + cp * cw * (Z - Z0) - cp * sw * (Y - Y0))


    i = np.arange(len(Tie_indices))
    df1_dX0 = (sp * (B1 + f) * ((Y - Y0) * t1 + (Z - Z0) * t4 + ck * cp * (X - X0))) / T2 ** 2 - (B2 * sp * T1) / T2 ** 2 - (ck * cp * (B1 + f)) / T2 - (B2 * cp * sk) / T2
    df2_dX0 = - (f * sp * T1) / T2 ** 2 - (f * cp * sk) / T2
    A[2 * i, camera_indices * 6] = df1_dX0
    A[2 * i + 1, camera_indices * 6] = df2_dX0

    df1_dY0 = (B2 * t2) / T2 - ((B1 + f) * t1) / T2 + (B2 * cp * sw * T1) / T2 ** 2 - (cp * sw * (B1 + f) * ((Y - Y0) * t1 + (Z - Z0) * t4 + ck * cp * (X - X0))) / T2 ** 2
    df2_dY0 = (f * t2) / T2 + (f * cp * sw * T1) / T2 ** 2
    A[2 * i, camera_indices * 6 + 1] = df1_dY0
    A[2 * i + 1, camera_indices * 6 + 1] = df2_dY0

    df1_dZ0 = (B2 * t3) / T2 - ((B1 + f) * t4) / T2 - (B2 * cp * cw * T1) / T2 ** 2 + (cp * cw * (B1 + f) * ((Y - Y0) * t1 + (Z - Z0) * t4 + ck * cp * (X - X0))) / T2 ** 2
    df2_dZ0 = (f * t3) / T2 - (f * cp * cw * T1) / T2 ** 2
    A[2 * i, camera_indices * 6 + 2] = df1_dZ0
    A[2 * i + 1, camera_indices * 6 + 2] = df2_dZ0

    df1_dw = (B2 * ((Y - Y0) * t3 - (Z - Z0) * t2)) / T2 - ((B1 + f) * ((Y - Y0) * t4 - (Z - Z0) * t1)) / T2 + ((B1 + f) * (cp * cw * (Y - Y0) + cp * sw * (Z - Z0)) * ((Y - Y0) * t1 + (Z - Z0) * t4 + ck * cp * (X - X0))) / T2 ** 2 - (
                B2 * (cp * cw * (Y - Y0) + cp * sw * (Z - Z0)) * T1) / T2 ** 2
    df2_dw = (f * ((Y - Y0) * t3 - (Z - Z0) * t2)) / T2 - (f * (cp * cw * (Y - Y0) + cp * sw * (Z - Z0)) * T1) / T2 ** 2
    A[2 * i, camera_indices * 6 + 3] = df1_dw
    A[2 * i + 1, camera_indices * 6 + 3] = df2_dw

    df1_dp = (B2 * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T1) / T2 ** 2 - (B2 * (sk * sp * (X - X0) + cp * cw * sk * (Z - Z0) - cp * sk * sw * (Y - Y0))) / T2 - ((B1 + f) * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * ((Y - Y0) * t1 + (Z - Z0) * t4 + ck * cp * (X - X0))) / (
                sp * (X - X0) + cp * cw * (Z - Z0) - cp * sw * (Y - Y0)) ** 2 - ((B1 + f) * (ck * sp * (X - X0) + ck * cp * cw * (Z - Z0) - ck * cp * sw * (Y - Y0))) / T2
    df2_dp = (f * (cp * (X - X0) - cw * sp * (Z - Z0) + sp * sw * (Y - Y0)) * T1) / T2 ** 2 - (f * (sk * sp * (X - X0) + cp * cw * sk * (Z - Z0) - cp * sk * sw * (Y - Y0))) / T2
    A[2 * i, camera_indices * 6 + 4] = df1_dp
    A[2 * i + 1, camera_indices * 6 + 4] = df2_dp

    df1_dk = ((B1 + f) * T1) / T2 + (B2 * ((Y - Y0) * t1 + (Z - Z0) * t4 + ck * cp * (X - X0))) / T2
    df2_dk = (f * ((Y - Y0) * t1 + (Z - Z0) * t4 + ck * cp * (X - X0))) / T2
    A[2 * i, camera_indices * 6 + 5] = df1_dk
    A[2 * i + 1, camera_indices * 6 + 5] = df2_dk

    df1_dX = (B2 * sp * T1) / T2 ** 2 - (sp * (B1 + f) * ((Y - Y0) * t1 + (Z - Z0) * t4 + ck * cp * (X - X0))) / T2 ** 2 + (ck * cp * (B1 + f)) / T2 + (B2 * cp * sk) / T2
    df2_dX = (f * sp * T1) / T2 ** 2 + (f * cp * sk) / T2
    A[2 * i, n_cameras * 6 + Tie_indices * 3] = df1_dX
    A[2 * i + 1, n_cameras * 6 + Tie_indices * 3] = df2_dX

    df1_dY = ((B1 + f) * t1) / T2 - (B2 * t2) / T2 - (B2 * cp * sw * T1) / T2 ** 2 + (cp * sw * (B1 + f) * ((Y - Y0) * t1 + (Z - Z0) * t4 + ck * cp * (X - X0))) / T2 ** 2
    df2_dY = - (f * t2) / T2 - (f * cp * sw * T1) / T2 ** 2
    A[2 * i, n_cameras * 6 + Tie_indices * 3 + 1] = df1_dY
    A[2 * i + 1, n_cameras * 6 + Tie_indices * 3 + 1] = df2_dY

    df1_dZ = ((B1 + f) * t4) / T2 - (B2 * t3) / T2 + (B2 * cp * cw * T1) / T2 ** 2 - (cp * cw * (B1 + f) * ((Y - Y0) * t1 + (Z - Z0) * t4 + ck * cp * (X - X0))) / T2 ** 2
    df2_dZ = (f * cp * cw * T1) / T2 ** 2 - (f * t3) / T2
    A[2 * i, n_cameras * 6 + Tie_indices * 3 + 2] = df1_dZ
    A[2 * i + 1, n_cameras * 6 + Tie_indices * 3 + 2] = df2_dZ


    i = np.arange(len(Tie_indices) * 2, len(Tie_indices) * 2 + len(Gcp_indices) * 3, 3)
    A[i, n_cameras * 6 + Gcp_indices * 3 + 0] = 1
    A[i + 1, n_cameras * 6 + Gcp_indices * 3 + 1] = 1
    A[i + 2, n_cameras * 6 + Gcp_indices * 3 + 2] = 1
    return A.tocsr()

class Bundle_information():
    def __init__(self,dir_files, gcp, TieXYZ, Observation_Tie, Observation_gcp, ncamera_params, IOP):
        # 5 -------------------- Mange Observation and Target Table --------------------
        Observation = np.row_stack((Observation_Tie, Observation_gcp))
        points_3d = np.row_stack((TieXYZ[:, 1:], gcp[:, 1:]))

        self.dir_files = dir_files
        self.Observation = Observation
        self.Observation_gcp = Observation_gcp
        self.Gcp_indices = gcp[:, 0].astype(int)
        self.Tie_indices = Observation[:, 1].astype(int)
        self.camera_indices = Observation[:, 0].astype(int)
        self.points_3d = np.row_stack((TieXYZ[:, 1:], gcp[:, 1:]))
        self.points_2d = Observation[:, 2:]
        self.n_cameras = len(ncamera_params)
        self.n_tiepoints = len(points_3d)
        self.n_gcpoints = len(gcp)
        self.GCP = gcp
        self.IOP = IOP

class SBA():
    def __init__(self, x0, Weight, Info_BA, Max_iter, th, prt, Show):
        print('  BA_loop    Max D_residual   Max D_Correction   RMSE_img(pix)      RMSE_Obj(m)        Phi         Time(h)')
        print('------------------------------------------------------------------------------------------------------------------')
        t0 = time.time()
        # 8 -------------------- Algorithm iterations --------------------
        DL, DX, Phi, DL_img, DL_obj, UV = [], [], [], [], [], []
        DL.append(0)
        DX.append(0)
        nTie, n_tiepoints, n_cameras, n_gcpoints = len(Info_BA.Tie_indices), Info_BA.n_tiepoints, Info_BA.n_cameras, len(Info_BA.Gcp_indices)
        n_cameras = Info_BA.n_cameras
        n_tiepoints = Info_BA.n_tiepoints
        camera_indices = Info_BA.camera_indices
        Tie_indices = Info_BA.Tie_indices
        Gcp_indices = Info_BA.Gcp_indices
        points_2d = Info_BA.points_2d
        GCP = Info_BA.GCP
        IOP = Info_BA.IOP

        for l in range(Max_iter):
            # 9 -------------------- Design Matrix and residuals --------------------
            A = Jac(x0, n_cameras, n_tiepoints, camera_indices, Tie_indices, Gcp_indices, IOP)
            dl, reproject = ObjFun(x0, n_cameras, n_tiepoints, camera_indices, Tie_indices, Gcp_indices, IOP, points_2d, GCP[:, 1:])

            # 11 -------------------- Apply Bundle Adjustment by RNE --------------------
            C = A[:, :n_cameras * 6]
            P = A[:, n_cameras * 6:]
            r = -(A.T @ Weight @ dl)
            rc = r[:n_cameras * 6]
            rp = r[n_cameras * 6:]

            # 9 -------------------- Normal Matrix -------------------
            U = C.T @ Weight @ C
            W = C.T @ Weight @ P
            Wt = P.T @ Weight @ C
            V = P.T @ Weight @ P
            iV = inverse_block_diag_sparse(V)

            # 10 -------------------- RNE --------------------
            AA = U - (W @ iV @ Wt)
            ll = rc - (W @ iV @ rp)
            dc = spsolve(AA, ll)
            dp = iV @ (rp - Wt @ dc)
            dx = np.hstack((dc, dp))
            x0 += dx

            # 12 -------------------- Save image and object residuals --------------------
            dl_img = (dl[:nTie * 2]).reshape(len(Tie_indices), 2)
            dl_obj = (dl[nTie * 2:]).reshape(len(Gcp_indices), 3)
            phi = (dl @ Weight @ dl.T) / (A.shape[0] - A.shape[1])
            DL.append(dl)
            DX.append(dx)
            UV.append(reproject)
            Phi.append(np.sqrt(np.mean(phi**2)))
            DL_img.append(np.sqrt(np.mean(dl_img ** 2, axis=0)))
            DL_obj.append(np.sqrt(np.mean(dl_obj ** 2, axis=0)))

            # 13 -------------------- Print image and object residuals --------------------
            if l == 0 or (l + 1) % prt == 0 or np.max(np.abs(DX[-2] - DX[-1])) < th or np.max(np.abs(DL[-2] - DL[-1])) < th:
                h = ((time.time() - t0) // 3600)
                min = np.round(((time.time() - t0) / 3600 - (time.time() - t0) // 3600) * 60)
                str1 = '  %4g         %5.2e         %5.2e' % (l + 1, np.max(np.abs(DL[-2] - DL[-1])), np.max(np.abs(DX[-2] - DX[-1])))
                str2 = '           %5.4f          %5.4f' % (np.sqrt(np.mean(dl_img ** 2)), np.sqrt(np.mean(dl_obj ** 2)))
                str3 = '       %4.4f      %3g:%2g' % (phi, h, min)
                print(str1+str2+str3)

            if np.max(np.abs(DX[-2] - DX[-1])) < th or np.max(np.abs(DL[-2] - DL[-1])) < th or l == Max_iter:
                Final_camera_params = x0[:n_cameras * 6].reshape((n_cameras, 6))
                Final_Tie = x0[n_cameras * 6:].reshape((n_tiepoints, 3))
                Final_Gcp = Final_Tie[Info_BA.Gcp_indices].reshape((n_gcpoints, 3))
                E = np.asarray(DL[-1]).reshape([Weight.shape[0], 1])
                break

        self.x_hat = x0
        self.e_hat = E
        self.Coordinate_point = Final_Tie
        self.Final_GCP = Final_Gcp
        self.A_matrix = A
        self.CameraParams = Final_camera_params
        self.Residuals = DL
        self.Corrections = DX
        self.Phi = np.asarray(Phi)
        self.Img_res = np.asarray(DL_img)
        self.Obj_res = np.asarray(DL_obj)

        if Show == True:
            PLOT_iter(np.asarray(DL_img), np.asarray(Phi), np.asarray(DL_obj))
            Plot_residual_hist(DL[1])
            Plot_residual_hist(DL[-1])
            show_residual(Info_BA.dir_files, Info_BA.Observation, UV[0], 3)
            show_residual(Info_BA.dir_files, Info_BA.Observation, UV[-1], 3)

