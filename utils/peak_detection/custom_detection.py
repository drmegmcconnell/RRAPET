"""
Custom Detection Template
"""
import numpy as np
# IMPORT FILTER COEFFICIENTS HERE
HH = [1, 2, 3, 4]


def my_filter(raw_data, hh=None):
    """
    REPLACE THIS CODE WITH YOUR OWN FILTER or hh-coefficients OR USE THOSE PROVIDED IN MHTD
    :param raw_data:
    :param hh:
    :return:
    """
    if hh is None:
        hh = HH
    filtered_data = np.convolve(raw_data, hh)
    filtered_data = filtered_data[int(len(hh) / 2):len(filtered_data) - (int(len(hh) / 2) + 1)]

    return filtered_data


def my_preprocessing_stage(filtered_data, var1=1, var2=1):
    """
    REPLACE THIS CODE WITH YOUR OWN PREPROCESSING SEGEMENT OR REMOVE
    For example, MHTD uses the HillTransform as a preprocessing tool
    :param filtered_data:
    :param var1:
    :param var2:
    :return:
    """
    pre_processed_data = filtered_data * var1 / var2

    return pre_processed_data


def my_QRS_detection(pre_processed_data, threshold=0):
    """
    Use this section to create or import a threshold and detect R_peaks based on decision rules
    For example, MHTD creates a variable threshold and uses the peak search algorithm here
    :param pre_processed_data:
    :param threshold:
    :return:
    """
    if threshold == 0:
        threshold = 10  # generate own threshold here or pass one to algorithm

    if pre_processed_data > threshold:
        R_peak_predictions = 1
    else:
        R_peak_predictions = 0

    return R_peak_predictions


def my_post_processing_tool(R_peak_predicitons, raw_data, rule1=0, rule2=1):
    """
    # Use this section to update R-peak predictions if required or remove if not required
    :param R_peak_predicitons:
    :param raw_data:
    :param rule1:
    :param rule2:
    :return:
    """
    corrected_R_peak_preds = R_peak_predicitons

    if R_peak_predicitons != rule1:
        corrected_R_peak_preds = 0
    elif R_peak_predicitons == rule2:
        corrected_R_peak_preds = 0

    pk_amplitude = raw_data[corrected_R_peak_preds]

    return corrected_R_peak_preds, pk_amplitude


def my_function(raw_data, hh=None):
    """
    # The following function is a compliation function which calls all of the segements one at a time to produce a set
    of time stamp and amplitude value R-peak predictions

    :param hh:
    :param raw_data:
    :return:
    """
    if hh is None:
        hh = HH
    filtered_data = my_filter(raw_data, hh)
    pre_processed_data = my_preprocessing_stage(filtered_data)
    R_peak_predictions = my_QRS_detection(pre_processed_data)
    corrected_R_peak_preds, pk_amplitude = my_post_processing_tool(R_peak_predictions, raw_data)

    return corrected_R_peak_preds, pk_amplitude
