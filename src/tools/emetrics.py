import numpy as np
from math import sqrt
from sklearn.metrics import average_precision_score
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def validate_inputs(*arrays):
    for array in arrays:
        if np.any(np.isnan(array)) or np.any(np.isinf(array)):
            raise ValueError("Input contains NaN or infinity values, which are not allowed.")

def get_aupr(Y, P, threshold=7.0):
    """
    Calculate the Area Under the Precision-Recall Curve (AUPR).
    
    Parameters:
        Y (np.array): True labels.
        P (np.array): Predicted scores.
        threshold (float): Threshold to binarize the labels.
    
    Returns:
        float: AUPR value.
    """
    validate_inputs(Y, P)
    Y_binary = np.where(Y >= threshold, 1, 0)
    P_binary = np.where(P >= threshold, 1, 0)
    aupr = average_precision_score(Y_binary, P_binary)
    logger.info(f'AUPR: {aupr}')
    return aupr

def get_cindex(Y, P):
    """
    Calculate the concordance index (C-index).
    
    Parameters:
        Y (np.array): True labels.
        P (np.array): Predicted scores.
    
    Returns:
        float: C-index value.
    """
    validate_inputs(Y, P)
    concordant = 0
    permissible = 0
    n = len(Y)
    for i in range(n):
        for j in range(i + 1, n):
            if Y[i] != Y[j]:
                permissible += 1
                if (Y[i] < Y[j] and P[i] < P[j]) or (Y[i] > Y[j] and P[i] > P[j]):
                    concordant += 1
                elif P[i] == P[j]:
                    concordant += 0.5
    cindex = concordant / permissible if permissible > 0 else 0
    logger.info(f'C-index: {cindex}')
    return cindex

def r_squared_error(y_obs, y_pred):
    """
    Calculate the R-squared error between observed and predicted values.
    
    Parameters:
        y_obs (np.array): Observed values.
        y_pred (np.array): Predicted values.
    
    Returns:
        float: R-squared error.
    """
    validate_inputs(y_obs, y_pred)
    ss_total = np.sum((y_obs - np.mean(y_obs)) ** 2)
    ss_residual = np.sum((y_obs - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    logger.info(f'R^2 Error: {r2}')
    return r2

def get_k(y_obs, y_pred):
    """
    Calculate the scaling factor k.
    
    Parameters:
        y_obs (np.array): Observed values.
        y_pred (np.array): Predicted values.
    
    Returns:
        float: Scaling factor k.
    """
    validate_inputs(y_obs, y_pred)
    k = np.dot(y_obs, y_pred) / float(np.dot(y_pred, y_pred))
    logger.debug(f'k value: {k}')
    return k

def squared_error_zero(y_obs, y_pred):
    """
    Calculate the R-squared error when the regression line passes through the origin.
    
    Parameters:
        y_obs (np.array): Observed values.
        y_pred (np.array): Predicted values.
    
    Returns:
        float: R-squared zero error.
    """
    validate_inputs(y_obs, y_pred)
    k = get_k(y_obs, y_pred)
    ss_total = np.sum((y_obs - np.mean(y_obs)) ** 2)
    ss_residual = np.sum((y_obs - k * y_pred) ** 2)
    r02 = 1 - (ss_residual / ss_total)
    logger.info(f'R^2_0 Error: {r02}')
    return r02

def get_rm2(ys_orig, ys_line):
    """
    Calculate the RM^2 metric.
    
    Parameters:
        ys_orig (np.array): Original observed values.
        ys_line (np.array): Predicted values.
    
    Returns:
        float: RM^2 value.
    """
    validate_inputs(ys_orig, ys_line)
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    rm2 = r2 * (1 - np.sqrt(np.absolute((r2 ** 2) - (r02 ** 2))))
    logger.info(f'RM^2: {rm2}')
    return rm2

def get_rmse(y, f):
    """
    Calculate the Root Mean Squared Error (RMSE).
    
    Parameters:
        y (np.array): Observed values.
        f (np.array): Predicted values.
    
    Returns:
        float: RMSE value.
    """
    validate_inputs(y, f)
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    logger.info(f'RMSE: {rmse}')
    return rmse

def get_mse(y, f):
    """
    Calculate the Mean Squared Error (MSE).
    
    Parameters:
        y (np.array): Observed values.
        f (np.array): Predicted values.
    
    Returns:
        float: MSE value.
    """
    validate_inputs(y, f)
    mse = ((y - f) ** 2).mean(axis=0)
    logger.info(f'MSE: {mse}')
    return mse

def get_pearson(y, f):
    """
    Calculate the Pearson correlation coefficient.
    
    Parameters:
        y (np.array): Observed values.
        f (np.array): Predicted values.
    
    Returns:
        float: Pearson correlation coefficient.
    """
    validate_inputs(y, f)
    rp = np.corrcoef(y, f)[0, 1]
    logger.info(f'Pearson Correlation: {rp}')
    return rp

def get_spearman(y, f):
    """
    Calculate the Spearman correlation coefficient.
    
    Parameters:
        y (np.array): Observed values.
        f (np.array): Predicted values.
    
    Returns:
        float: Spearman correlation coefficient.
    """
    validate_inputs(y, f)
    rs = stats.spearmanr(y, f)[0]
    logger.info(f'Spearman Correlation: {rs}')
    return rs

def get_ci(y, f):
    """
    Calculate the concordance index (CI).
    
    Parameters:
        y (np.array): Observed values.
        f (np.array): Predicted values.
    
    Returns:
        float: CI value.
    """
    validate_inputs(y, f)
    sorted_indices = np.argsort(y)
    y_sorted = y[sorted_indices]
    f_sorted = f[sorted_indices]

    concordant = 0
    permissible = 0
    n = len(y_sorted)
    for i in range(n):
        for j in range(i + 1, n):
            if y_sorted[i] != y_sorted[j]:
                permissible += 1
                if (y_sorted[i] < y_sorted[j] and f_sorted[i] < f_sorted[j]) or (y_sorted[i] > y_sorted[j] and f_sorted[i] > f_sorted[j]):
                    concordant += 1
                elif f_sorted[i] == f_sorted[j]:
                    concordant += 0.5
    ci = concordant / permissible if permissible > 0 else 0
    logger.info(f'CI: {ci}')
    return ci