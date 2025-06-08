import numpy as np
from INFtoCOV import inf_to_cov
from Mupdate import mupdate
from Tupdate import tupdate

def idpkf(k, Z, w, u, X, V, R, H, Phi, gamma, Qk, Form, h=None):
    """
    Apply Influence Diagram Probabilistic Kalman filter at time k.

    Parameters:
    k (int): Desired discrete time.
    Z (numpy.ndarray): list of measurements [z1, z2, ..., zm], each of shape (dim_z,1)
    w (numpy.ndarray): list of weights corresponding to Z_list
    u (numpy.ndarray): An n x 1 vector representing the mean of the state at time k.
    X (numpy.ndarray): If k=0, the covariance matrix of state at time k. If k≠0, the matrix of Gaussian influence diagram arc coefficients.
    V (numpy.ndarray): If k≠0, an n x 1 vector of Gaussian influence diagram conditional variances. Ignored if k=0.
    R (numpy.ndarray): The measurement noise covariance matrix R.ww
    H (numpy.ndarray): The measurement matrix at discrete time k.
    Phi (numpy.ndarray): The state transition matrix at time k.
    gamma (numpy.ndarray): The process noise matrix at time k.
    Qk (numpy.ndarray): The process noise covariance matrix.
    Form (int): Determines the output form (0 for ID form, 1 for covariance form).

    Returns:
    u (numpy.ndarray): The updated mean vector.
    B (numpy.ndarray): The updated state covariance matrix or influence diagram form matrix.
    V (numpy.ndarray): The updated vector of conditional variances.
    """

    # Get dimensions
    domain = X.shape[0]
    dim_z = Z[0].shape[0]
    m = len(Z)

    # Create stacked measurement vector
    Z_exp = np.vstack(Z)

    # Create expanded H
    H_exp = np.kron(np.ones((m, 1)), H)

    # Create expanded R matrix: block diagonal with R / w_i
    R_exp = np.zeros((m * dim_z, m * dim_z))
    for i in range(m):
        R_block = R / w[i]
        R_exp[i * dim_z:(i+1) * dim_z, i * dim_z:(i+1) * dim_z] = R_block

    # Perform measurement update
    u, V, B = mupdate(k, Z_exp, u, X, V, R_exp, H_exp, h)
    u_new = u[:domain]
    V_new = V[:domain]
    B_new = B[:domain, :domain]

    # Perform time update
    u, B, V = tupdate(u_new, B_new, V_new, Phi, gamma, Qk)

    # Convert back to covariance form if required
    if Form == 1:
        B = inf_to_cov(V, B, domain)

    return u, B, V

