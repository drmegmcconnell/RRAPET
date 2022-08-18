"""
R-peak Detection Methods
K-means

"""
import numpy as np
from .detector import detect_peaks
from scipy.cluster.vq import kmeans2
from scipy.interpolate import pchip_interpolate


def _mean_filter(data, width):
    data = np.reshape(data, [1, np.size(data)])

    [f, c] = np.shape(data)

    out = np.zeros([f, c])
    for i in range(f):
        indice = 0
        for j in range(int(width - np.floor(width / 2)), width + 1):
            out[i, indice] = np.sum(data[0, 0: j]) / j
            indice += 1

    for i in range(f):
        indice = int(np.floor(width / 2) + 1)
        for j in range(width, c):
            out[i, indice] = ((out[i, indice - 1] * width) - data[i, j - width] + data[i, j]) / width
            indice += 1

    for i in range(f):

        indice = int(c - (width - np.floor(width / 2)) + 1)
        for j in range(0, int(width - np.floor(width / 2) - 1)):
            out[i, indice] = sum(data[i, c - width + j + 1: c]) / (width - j - 1)
            indice += 1

    return out


def _envelopment_filter(data, orden):
    # Error Checking
    error_checking = 0

    data = np.reshape(data, [1, np.size(data)])

    if error_checking == 1:
        if data.dtype != 'float64' or data.dtype != 'float32':
            print('Error - check that data input is numeric and real')

        (a, b) = data.shape

        if a > 1 & b > 1:
            print('Error - check that data input is a vector')

        if orden.dtype != 'int64' or data.dtype != 'int32':
            print('Error - check that orden input is numeric and an integer')

        if orden <= 0:
            print('Orden needs to be a real positive integer')

    out = np.zeros(data.shape)

    for i in range(orden):
        envI = _lowEnvelopment(data)
        envI2 = _lowEnvelopment(envI)

        out = out + (envI + envI2) / 2

        data = data - out

    return out


def _lowEnvelopment(din):  # Only does pchip interpolation

    if len(din) == 1:
        (a, b) = din.shape
        if a > 1 & b > 1:
            print('Error - check that data input is a vector')

    # Missing an error check here to ensure interpolation type is a real type
    data = din[:]
    d = np.diff([data])
    if len(d) == 1:
        d = np.reshape(d, [np.size(d), ])
    N = len(d)

    if N == 0:
        env = data
    else:

        extrema = np.where(np.multiply(d[1:N], d[0:N - 1]) <= 0)
        if len(extrema) == 1:
            extrema = np.reshape(extrema, [np.size(extrema)])
        localMini = [0]
        localMini = np.append(localMini, extrema[np.where(d[extrema] < 0)] + 1)
        localMini = np.append(localMini, N)

        if len(data) == 1:
            data = np.reshape(data, [np.size(data)])

        env = pchip_interpolate(localMini, data[localMini], np.arange(0, N + 1))

    return env


def ECG_processing(ecg):
    """

    :param ecg: ECG waveform as 1D signal (assumes 1 lead)
    :return: R-peak Locations
    """
    ecg2 = _mean_filter(ecg[:], 3)
    qrs = ecg2[:] - _envelopment_filter(ecg2[:], 2)
    qrs = _mean_filter(qrs, 3)
    qrs = qrs - _envelopment_filter(qrs[:], 2)
    qrs = np.reshape(qrs, [np.size(qrs), ])
    index = detect_peaks(qrs)

    localMaximums = qrs[index]

    # nonQRS_Labels = KMeans(n_clusters=2, random_state=0).fit(localMaximums)

    nonQRS_Labels = kmeans2(localMaximums, 2, minit='random')

    QRS_Labels = np.ones(len(nonQRS_Labels[1]), dtype='int')
    QRS_Labels = (QRS_Labels != nonQRS_Labels[1])

    nonQRSvalues = localMaximums * nonQRS_Labels[1]
    QRSvalues = localMaximums * QRS_Labels

    QRSvalues = QRSvalues[QRSvalues != 0]
    nonQRSvalues = nonQRSvalues[nonQRSvalues != 0]
    nonQRS_Labels = nonQRS_Labels[nonQRS_Labels != 0]
    QRS_Labels = QRS_Labels[QRS_Labels != 0]

    if np.min(QRSvalues) < np.max(nonQRSvalues):
        QRS_Labels = nonQRS_Labels

    if np.shape(QRS_Labels) == (2,):
        pos = index[QRS_Labels[1][:] > 0.0]
    else:
        pos = index[QRS_Labels[:]]

    i = 1

    while i < len(pos):
        df = np.diff(qrs[np.arange(int(pos[i - 1]), int(pos[i] + 1))])

        sdf = df[np.arange(len(df) - 1)] * df[np.arange(1, len(df))]
        sdf = np.append(sdf, 0)

        if np.sum(df[sdf < 0] < 0) == 1:
            pos[i] = -1
            i = i + 1

        i = i + 1

    pos = pos[pos != -1]

    return pos
