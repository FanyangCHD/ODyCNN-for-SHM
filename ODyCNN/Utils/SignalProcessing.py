# -*-coding:utf-8-*-
"""
Created on 2022.3.5
programing language:python
@author:夜剑听雨
"""
import numpy as np
import math
from scipy import signal
import torch

def compare_SNR(recov_img, real_img):
    """
    计算信噪比
    :param recov_img:重建后或者含有噪声的数据
    :param real_img: 干净的数据
    :return: 信噪比
    """

    real = np.linalg.norm(real_img, ord='fro')

    noise = np.linalg.norm(real_img - recov_img, ord='fro')

    if noise == 0 or real==0:
      s = 999.99
    else:
      s = 10*math.log(real/noise, 10)
    return s

def batch_snr(de_data, clean_data):
    """
    计算一个batch的平均信噪比
    :param de_data: 去噪后的数据
    :param clean_data: 干净的数据
    :return: 一个batch的平均信噪比
    """
    De_data = de_data.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    Clean_data = clean_data.data.cpu().numpy()
    SNR = 0
    for i in range(De_data.shape[0]):
        De = De_data[i, :, :, :].squeeze()  # 默认压缩所有为1的维度
        Clean = Clean_data[i, :, :, :].squeeze()
        SNR += compare_SNR(De, Clean)
    return SNR / De_data.shape[0]

def mask_mse(y_pred, y_true, mask):
    
    diff = torch.mul(mask, (y_pred - y_true))
    reshaped_tensor = diff.view(diff.shape[0], -1)  # 将张量形状变为 [bs, c*h]

    # 计算矩阵的二范数的平方
    norm_squared = torch.norm(reshaped_tensor, p=2, dim=1)**2  # 在 dim=1 上计算二范数的平方，得到形状为 [bs]

    # 求所有矩阵的和
    sum_norm_squared = torch.sum(norm_squared)

    # 除以 bs
    result = sum_norm_squared / diff.shape[0]

    # squared_norm = torch.norm(diff, p='fro', dim=(1, 2)) ** 2
    # mean_squared_norm = torch.mean(squared_norm)
    return result


def error1(y_pred,y_true):

    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    diff = y_true - y_pred
    num = np.linalg.norm(diff, ord=2)
    den = np.linalg.norm(y_true, ord=2)
    error = num / den
    return error

def mse1(y_pred,y_true):
    return 10000*np.mean((y_pred - y_true) ** 2)

def rmse1(y_pred,y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def r_squared1(y_pred,y_true):
    """
    计算 R² (决定系数)
    :param y_true: 真实值向量
    :param y_pred: 预测值向量
    :return: R² 值
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  # 总平方和
    ss_residual = np.sum((y_true - y_pred) ** 2)         # 残差平方和
    return 1 - (ss_residual / ss_total)

def calculate_error(y_pred,y_true):
    """
    计算误差（error）

    参数:
    - y_true: 真实值的数组或列表
    - y_pred: 预测值的数组或列表

    返回值:
    - mse: 均方误差值
    """
    y_pred = y_pred.data.cpu().numpy() 
    y_true = y_true.data.cpu().numpy() 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num1 = np.sum((y_true - y_pred) ** 2)
    num = np.sqrt(num1)
    den1 = np.sum(y_true ** 2)
    den = np.sqrt(den1)
    return num / den

def r_squared(y_pred, y_true):
    """
    计算R平方

    参数:
        y_true: 实际观测到的因变量的值 (列表或数组)
        y_pred: 回归模型预测的因变量的值 (列表或数组)

    返回值:
        r_squared: R平方值 (float)
    """
    y_pred = y_pred.data.cpu().numpy() 
    y_true = y_true.data.cpu().numpy() 
    mean_true = sum(y_true) / len(y_true)
    total_sum_squares = sum((y - mean_true) ** 2 for y in y_true)
    residual_sum_squares = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    
    return r_squared

def calculate_rmse(y_pred,y_true):
    """
    计算均方误差（Mean Squared Error，MSE）

    参数:
    - y_true: 真实值的数组或列表
    - y_pred: 预测值的数组或列表

    返回值:
    - mse: 均方误差值
    """
    y_pred = y_pred.data.cpu().numpy() 
    y_true = y_true.data.cpu().numpy() 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = math.sqrt(mse)
    return rmse

# --------------------------------------------
# PSNR
# --------------------------------------------


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    else:
        return 10 * np.log10(4 / mse)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    img2 = img2.data.cpu().numpy()
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    return psnr(img1, img2)

# --------------------------------------------
# --------------------------------------------


def fft_spectrum(Signal, SampleRate):
    """
    计算一维信号的傅里叶谱
    :param Signal: 一维信号
    :param SampleRate: 采样率，一秒内的采样点数
    :return: 傅里叶变换结果
    """
    fft_len = Signal.size  # 傅里叶变换长度
    # 原函数值的序列经过快速傅里叶变换得到一个复数数组，复数的模代表的是振幅，复数的辐角代表初相位
    SignalFFT = np.fft.rfft(Signal) / fft_len  # 变换后归一化处理
    SignalFreqs = np.linspace(0, SampleRate/2, int(fft_len/2)+1)  # 生成频率区间
    SignalAmplitude = np.abs(SignalFFT) * 2   # 复数的模代表的是振幅
    return SignalFreqs, SignalAmplitude

# 巴沃斯低通滤波器
def butter_lowpass(cutoff, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    normal_cutoff = cutoff / rate
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(noise_data, cutoff, sample_rate, order=4):
    """
    低通滤波器
    :param noise_data: 含噪声数据
    :param cutoff: 低通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_lowpass(cutoff, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

# 巴沃斯带通滤波器
def butter_bandpass(lowcut, highcut, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    low = lowcut / rate
    high = highcut / rate
    b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)
    return b, a

def bandpass_filter(noise_data, lowcut, highcut, sample_rate, order=4):
    """
    带通滤波器
    :param noise_data: 含噪声数据
    :param lowcut: 带通滤波的最小值
    :param higtcut: 带通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data
# 巴沃斯高通滤波器
def butter_highpass(cutup, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    normal_cutup = cutup / rate
    b, a = signal.butter(order, normal_cutup, btype='high', analog=False)
    return b, a

def highpass_filter(noise_data, cutup, sample_rate, order=4):
    """
    低通滤波器
    :param noise_data: 含噪声数据
    :param cutoff: 低通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_highpass(cutup, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

# 一维信号的中值滤波器
# python的中值滤波函数对数组的维数要求严格，打个比方你用维数为（200，1）的数组当输入，不行！
# 必须改成（200，才会给你滤波。
def mide_filter(x,kernel_size=5):
    """
    中值滤波器
    :param x: 一维信号
    :param kernel_size: 滤波器窗口，默认为5
    :return: 中值滤波后的数据
    """
    x1 = x.reshape(x.size)
    y = signal.medfilt(x1, kernel_size=kernel_size)
    return y

def fk_spectra(data, dt, dx, L=6):
    """
    f-k(频率-波数)频谱分析
    :param data: 二维的地震数据
    :param dt: 时间采样间隔
    :param dx: 道间距
    :param L: 平滑窗口
    :return: S(频谱结果), f(频率范围), k(波数范围)
    """
    data = np.array(data)
    [nt, nx] = data.shape  # 获取数据维度
    # 计算nk和nf是为了加快傅里叶变换速度,等同于nextpow2
    i = 0
    while (2 ** i) <= nx:
        i = i + 1
    nk = 4 * 2 ** i
    j = 0
    while (2 ** j) <= nt:
        j = j + 1
    nf = 4 * 2 ** j
    S = np.fft.fftshift(abs(np.fft.fft2(data, (nf, nk))))  # 二维傅里叶变换
    H1 = np.hamming(L)
    # 设置汉明窗口大小，汉明窗的时域波形两端不能到零，而海宁窗时域信号两端是零。从频域响应来看，汉明窗能够减少很近的旁瓣泄露
    H = (H1.reshape(L, -1)) * (H1.reshape(1, L))
    S = signal.convolve2d(S, H, boundary='symm', mode='same')  # 汉明平滑
    S = S[nf // 2:nf, :]
    f = np.arange(0, nf / 2, 1)
    f = f / nf / dt
    k = np.arange(-nk / 2, nk / 2, 1)
    k = k / nk / dx
    return S, k, f


