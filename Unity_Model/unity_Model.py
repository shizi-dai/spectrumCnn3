import pickle

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置默认字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

import tensorflow as tf
from scipy.optimize import curve_fit
from tensorflow.keras.models import load_model


# 定义用于拟合的洛伦兹函数（与之前的区别在于增加了幅度变量A）
# 定义洛伦兹函数
def lorentzian(x, *params):
    x0 = params[0]
    gamma = params[1]
    lambda_var = params[2]
    A = 1.
    return A - A / (1. / lambda_var + ((x - x0) / gamma) ** 2)


def lorentzian2(x, *params):
    x0 = params[0:2]
    gamma = params[2:4]
    lambda_var = params[4:6]
    A = 1. / 2.

    param1 = [x0[0], gamma[0], lambda_var[0]]
    param2 = [x0[1], gamma[1], lambda_var[1]]
    param = [param1, param2]

    signal = 0
    for i in range(0, 2):
        signal += A * lorentzian(x, *param[i])
    return signal


def lorentzian4(x, *params):
    x0 = params[0:4]
    gamma = params[4:8]
    lambda_var = params[8:12]
    A = 1. / 4.

    param1 = [x0[0], gamma[0], lambda_var[0]]
    param2 = [x0[1], gamma[1], lambda_var[1]]
    param3 = [x0[2], gamma[2], lambda_var[2]]
    param4 = [x0[3], gamma[3], lambda_var[3]]

    param = [param1, param2, param3, param4]

    signal = 0
    for i in range(0, 4):
        signal += A * lorentzian(x, *param[i])
    return signal


def LorentzianFit(x_range_fit, spectrum_true, x_range, peakNum, paramInitGuess, error_num):
    # 初始猜测值
    initial_guess = paramInitGuess

    try:
        if peakNum == 1:
            fit_param, covariance = curve_fit(lorentzian, x_range_fit, spectrum_true, p0=initial_guess,
                                              maxfev=20000)
            fit_spectrum = lorentzian(x_range, *fit_param)
            fit_spectrum_eval = lorentzian(x_range_fit, *fit_param)
        elif peakNum == 2:
            fit_param, covariance = curve_fit(lorentzian2, x_range_fit, spectrum_true, p0=initial_guess,
                                              maxfev=20000)
            fit_spectrum = lorentzian2(x_range, *fit_param)
            fit_spectrum_eval = lorentzian2(x_range_fit, *fit_param)
        elif peakNum == 4:
            fit_param, covariance = curve_fit(lorentzian4, x_range_fit, spectrum_true, p0=initial_guess,
                                              maxfev=20000)
            fit_spectrum = lorentzian4(x_range, *fit_param)
            fit_spectrum_eval = lorentzian4(x_range_fit, *fit_param)
        else:
            covariance = [0]
            fit_param = [0]
            fit_spectrum = np.ones_like(x_range)
            fit_spectrum_eval = np.ones_like(x_range_fit)
        error_num = error_num + 0
    except (RuntimeError, ValueError) as e:
        print('fit error')
        error_num = error_num + 1
        fit_param = [0]
        fit_spectrum = np.ones_like(x_range)
        fit_spectrum_eval = np.ones_like(x_range_fit)
        covariance = [0]

    spectrum_true = np.array(spectrum_true)

    # 计算均方误差
    fit_mse = np.mean((fit_spectrum_eval - spectrum_true) ** 2)

    # 计算总平方和 (SST)
    fit_sst = np.sum((spectrum_true - np.mean(spectrum_true)) ** 2)

    # 计算残差平方和 (SSE)
    fit_sse = np.sum((spectrum_true - fit_spectrum_eval) ** 2)

    # 计算决定系数 (R²)
    r2 = 1 - (fit_sse / fit_sst)

    return fit_spectrum, fit_param, error_num, covariance, fit_mse, r2


# 定义编码器，将二维数据映射到（0,1），返回最终的归一数据以及归一化参数
def encoder(data):
    data = np.array(data)
    normalized_max = data.max(axis=1, keepdims=True)
    normalized_min = data.min(axis=1, keepdims=True)
    data_normalized = (data - normalized_min) / (normalized_max - normalized_min)
    return data_normalized, [normalized_max, normalized_min]


# 定义解码器，将编码器归一化的数据通过其参数返回为原始数据
def decoder(data_normalized, normalized_param):
    data_normalized = np.array(data_normalized)
    normalized_max = normalized_param[0]
    normalized_min = normalized_param[1]
    data = data_normalized * (normalized_max - normalized_min) + normalized_min
    return data


def uniformSampling(spectrum_data_nomalized, sample_interval=[4]):
    """
    对输入数据进行均匀取样，每隔 sample_interval 个数据取一次样，未取样到的点设为1。

    参数：
    spectrum_data_nomalized: 输入的二维数组
    sample_interval: 取样间隔，默认为2

    返回：
    input_data_array: 输入数据的NumPy数组
    output_data_array: 输出数据的NumPy数组
    """
    # 初始化空的输入数据和输出数据列表
    input_data_list = []
    output_data_list = []

    # 对每个取样间隔进行操作
    for interval in sample_interval:
        # 对每个一维列表执行操作
        for spectrum in spectrum_data_nomalized:
            # 复制spectrum以保留原始数据
            input_data = np.copy(spectrum)

            # 对每个位置进行操作
            for i in range(len(spectrum)):
                # 如果位置不是取样点，则设为1
                if i % interval != 0:
                    input_data[i] = 1

            # 合并数据到列表中
            input_data_list.append(input_data)
            output_data_list.append(spectrum)

    # 将列表转换为NumPy数组
    input_data_array = np.array(input_data_list)
    output_data_array = np.array(output_data_list)

    # 打印结果
    print("Input Data Shape:", input_data_array.shape)
    print("Output Data Shape:", output_data_array.shape)

    return input_data_array, output_data_array


'''
-------------------说明-------------------
分类器和谱线还原模型使用归一化编码，
预测拟合参数模型未使用归一化编码（因为归一化会丢失信息）

'''

# -------------------------准备数据---------------------------
spectrum_data = []  # 原始未归一化、未采样谱线（n，202）
x_range = []
# 导入数据
dir_path = '../DataGenerater/lorentzianSpectrum_202_contrast/validateData/'
with open(dir_path + 'spectrum_data_normalized.pickle', 'rb') as spectrum_data_normalized_file:
    spectrum_data = pickle.load(spectrum_data_normalized_file)
with open(dir_path + 'x_range.pickle', 'rb') as x_range_file:
    x_range = pickle.load(x_range_file)
    x_range = np.array(x_range)
with open(dir_path + 'label_data.pickle', 'rb') as label_param_data:
    label_params = pickle.load(label_param_data)  # list

# 归一化编码
spectrum_data_normalized, spectrum_data_normalizedParam = encoder(spectrum_data)
# 均匀采样
# ****注意，下面的均匀采样函数仅仅使用一种间隔输入，这样方便查看和分析*****
# ****若使用两种间隔混合，则归一化参数需要调整
spectrum_data_normalized_uniformSampling_input, spectrum_data_normalized_uniformSampling_output = \
    uniformSampling(spectrum_data_normalized)
# 准备输入数据
spectrum_input = spectrum_data_normalized_uniformSampling_input

# 一些参数
spectrum_length = spectrum_input.shape[0]
# -------------------导入模型-----------------------------
# classifier (分类器)
dir_name_classifier = '../Classifier/Model_contrast/classifier.keras'
model_classifier = load_model(dir_name_classifier)

# models that are used to predict half-peak-width
# network/SpectrumOutputNetwork/conv1D_fit_withFullSpectrum/separater/Model_contrast/DoublePeak/model_fitParams_halPeWid_contrast.keras
dir_name_fit_params_model = '../conv1D_fit_withFullSpectrum/separater/Model_contrast/'
model_fitParams_halPeWid_contrast_doublePeak = load_model(
    dir_name_fit_params_model + 'DoublePeak/model_fitParams_halPeWid_contrast.keras')
model_fitParams_halPeWid_contrast_singlePeak = load_model(
    dir_name_fit_params_model + 'SinglePeak/model_fitParams_halPeWid_contrast.keras')
model_fitParams_halPeWid_contrast_fourPeak = load_model(
    dir_name_fit_params_model + 'FourPeak/model_fitParams_halPeWid_contrast.keras')
# 导入其放缩因子
with open(dir_name_fit_params_model + 'DoublePeak/label_factor_expand.pickle', 'rb') as label_factor_doublePeak_file:
    label_factor_doublePeak = np.array(pickle.load(label_factor_doublePeak_file))
with open(dir_name_fit_params_model + 'SinglePeak/label_factor_expand.pickle', 'rb') as label_factor_singlePeak_file:
    label_factor_singlePeak = np.array(pickle.load(label_factor_singlePeak_file))
with open(dir_name_fit_params_model + 'FourPeak/label_factor_expand.pickle', 'rb') as label_factor_fourPeak_file:
    label_factor_fourPeak = np.array(pickle.load(label_factor_fourPeak_file))
# models that used to predict peak position
model_fitParams_peakPosition_doublePeak = load_model(
    dir_name_fit_params_model + 'DoublePeak/model_fitParams_peakPosition.keras')
model_fitParams_peakPosition_singlePeak = load_model(
    dir_name_fit_params_model + 'SinglePeak/model_fitParams_peakPosition.keras')
model_fitParams_peakPosition_fourPeak = load_model(
    dir_name_fit_params_model + 'FourPeak/model_fitParams_peakPosition.keras')

# model that are used to predict spectrum
dir_name_spectrum_model = '../conv1D_spectrum_no_fit/Model_contrast/'
model_spectrum_doublePeak = load_model(dir_name_spectrum_model + 'DoublePeak/model_spectrum.keras')
model_spectrum_singlePeak = load_model(dir_name_spectrum_model + 'SinglePeak/model_spectrum.keras')
model_spectrum_fourPeak = load_model(dir_name_spectrum_model + 'FourPeak/model_spectrum.keras')

# --------------------整合模型------------------------------------------------------------------------------
# 分类器分类
peak_classify = model_classifier(spectrum_input)  # one-hot编码形式的输出,例如[0,0,0,1]表示4峰
peak_classify = np.array(peak_classify)
# 获取最大值所在的索引，然后加一，就是峰的数量
peakNums = np.argmax(peak_classify, axis=1) + 1

# 预测谱线--------------------------
model_predict_spectrum = []
for i in range(0, len(peakNums)):
    peakNum = peakNums[i]
    if peakNum == 1:
        predict_spectrum = model_spectrum_singlePeak(np.reshape(spectrum_input[i], [1, spectrum_input.shape[1]]))
    elif peakNum == 2:
        predict_spectrum = model_spectrum_doublePeak(np.reshape(spectrum_input[i], [1, spectrum_input.shape[1]]))
    elif peakNum == 4:
        predict_spectrum = model_spectrum_fourPeak(np.reshape(spectrum_input[i], [1, spectrum_input.shape[1]]))
    else:
        predict_spectrum = np.ones_like(spectrum_input[i])
        print('classifier error')
    predict_spectrum = np.reshape(np.array(predict_spectrum), [202, ])
    model_predict_spectrum.append(predict_spectrum)
# 解码还原
model_predict_spectrum = decoder(model_predict_spectrum, spectrum_data_normalizedParam)  # 解码还原
model_predict_spectrum_normalized, model_predict_spectrum_normalizedParams = encoder(model_predict_spectrum)  # 归一化
# 拟合谱线---------------------------
# 拟合参数模型的输入为谱预测模型的预测谱（未归一化的数据）
# 拟合使用的数据为原始采样数据
# 将输入谱线解码还原
spectrum_input_True = decoder(spectrum_input, spectrum_data_normalizedParam)

model_predict_fitParams = []
fit_spectrums = []
fitParams = []
covariances = []  # 保存拟合协方差矩阵
error_num = 0  # 计数拟合错误率
fit_mse_array = []  # 保存拟合的均方误差
r2_array = []  # 保存拟合的决定系数R2

for i in range(0, len(peakNums)):
    peakNum = peakNums[i]  # 峰的数量
    # 预测拟合参数
    if peakNum == 1:
        predict_peakPosition = model_fitParams_peakPosition_singlePeak(
            np.reshape(model_predict_spectrum_normalized[i], [1, model_predict_spectrum_normalized.shape[1]]))
        predict_halPeWid_contrast = model_fitParams_halPeWid_contrast_singlePeak(
            np.reshape(model_predict_spectrum[i], [1, model_predict_spectrum.shape[1]]))
        print(predict_halPeWid_contrast.shape)
        predict_fitParams = np.concatenate((predict_peakPosition, predict_halPeWid_contrast), axis=1)
        predict_fitParams = predict_fitParams / label_factor_singlePeak
    elif peakNum == 2:
        predict_peakPosition = model_fitParams_peakPosition_doublePeak(
            np.reshape(model_predict_spectrum_normalized[i], [1, model_predict_spectrum_normalized.shape[1]]))
        predict_halPeWid_contrast = model_fitParams_halPeWid_contrast_doublePeak(
            np.reshape(model_predict_spectrum[i], [1, model_predict_spectrum.shape[1]]))
        predict_fitParams = np.concatenate((predict_peakPosition, predict_halPeWid_contrast), axis=1)
        predict_fitParams = predict_fitParams / label_factor_doublePeak
    elif peakNum == 4:
        predict_peakPosition = model_fitParams_peakPosition_fourPeak(
            np.reshape(model_predict_spectrum_normalized[i], [1, model_predict_spectrum_normalized.shape[1]]))
        predict_halPeWid_contrast = model_fitParams_halPeWid_contrast_fourPeak(
            np.reshape(model_predict_spectrum[i], [1, model_predict_spectrum.shape[1]]))
        predict_fitParams = np.concatenate((predict_peakPosition, predict_halPeWid_contrast), axis=1)
        predict_fitParams = predict_fitParams / label_factor_fourPeak
    else:
        predict_fitParams = [0]
        print('classifier error')
    model_predict_fitParams.append(predict_fitParams)
    # 由预测拟合参数作为初始值进行拟合
    if (predict_fitParams != [0]).any():
        if peakNum == 0:
            fit_spectrum, fitParam, error_num, covariance, fit_mse, r2 = LorentzianFit(x_range,
                                                                                       model_predict_spectrum[i],
                                                                                       x_range,
                                                                                       peakNum,
                                                                                       predict_fitParams, error_num)
        else:
            # 解码还原，然后获取被采样到的点的索引,
            # 将这些非1的谱点取出用于拟合
            spectrum_true = spectrum_input_True[i]
            non_one_indices = np.where(spectrum_input[i] != 1)

            spectrum_true_fit = spectrum_true[non_one_indices]
            x_range_fit = x_range[non_one_indices]
            fit_spectrum, fitParam, error_num, covariance, fit_mse, r2 = LorentzianFit(x_range_fit, spectrum_true_fit,
                                                                                       x_range,
                                                                                       peakNum, predict_fitParams,
                                                                                       error_num)
    else:
        fit_spectrum = np.repeat(spectrum_input_True[i].min(), spectrum_input.shape[1])
        fitParam = [0]
        fit_mse = 1
        r2 = 0
        error_num = error_num + 1
        print('classifier error')

    fit_spectrums.append(fit_spectrum)
    fitParams.append(fitParam)
    fit_mse_array.append(fit_mse)
    r2_array.append(r2)

# --------------------可视化结果，对比分析----------------------------------------
'''
*原始数据：spectrum_data
归一化数据：spectrum_data_normalized
归一化参数：spectrum_data_normalizedParam
*归一化的输入数据：spectrum_input（对归一化数据均匀采样后得到的）
*还原的输入数据：specturm_input_true (但是spectrum_input中的1被还原为了最大值)


分类器输出数据：peak_classify（one-hot编码）
预测峰数量数据：peakNums

*预测谱数据：model_predict_spectrum

预测参数数据：model_predict_fitParams
拟合参数数据：fitParams
*拟合谱线数据：fit_spectrums
'''

# 选择num个样本进行对比
num = 50
sample_indices = []
for i in range(num):
    sample_indices.append(int(i * len(spectrum_data) / (num + 1)))

director_name = 'results/'  # 结果存储目录
# 画出num张对比图
for index in sample_indices:
    plt.figure(figsize=(12, 6))

    # 输入谱线
    plt.scatter(x_range, spectrum_input_True[index], label='Input Spectrum')

    # 真实谱线
    plt.plot(x_range, spectrum_data[index], label='True Output Spectrum', linestyle='--', alpha=0.7)

    # 模型预测谱线
    plt.plot(x_range, model_predict_spectrum[index],
             label='Model Predicted Spectrum', linestyle='--')

    # 洛伦兹函数拟合谱线
    plt.plot(x_range, fit_spectrums[index], label='Lorentzian fit Spectrum', linestyle='--')

    # classifier分类情况
    plt.scatter(x_range[0], np.array(spectrum_data[index]).min(), label='classifier predict: ' + str(peakNums[index]))
    plt.scatter(x_range[0], np.array(spectrum_data[index]).max(), label='True peakNum')

    # 对比中心位置、半峰宽的预测情况
    # print('True params', test_output_data_array[index][0:out_shape_2_4])
    # print('Predicted params', model_predict_data[index][0:out_shape_2_4], '\n')

    # 打印真实参数与预测参数
    print(f"sample{index}")
    print("真实参数: ", label_params[index])
    print("拟合参数: ", fitParams[index])
    print('均方误差：', fit_mse_array[index])
    print('决定系数：', r2_array[index])

    plt.title(f'Spectrum Comparison (Sample {index + 1})')
    plt.xlabel('frequency(GHz)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig(director_name + f'figure_sample{index + 1}.png')
# plt.show()

mse_num = 0  # 均方差低于设定值的计数和比例
for i in fit_mse_array:
    if i <= 5e-5:
        mse_num = mse_num + 1

r2_num = 0  # 拟合优度高于设定值的计数和比例
for i in r2_array:
    if i > 0.98:
        r2_num = r2_num + 1

print('拟合错误率：', error_num / spectrum_input.shape[0])
print('平均均方误差', np.mean(fit_mse_array))
print('MSE低于5e-5的样本数：', mse_num)
print('MSE低于5e-5的样本比例：', mse_num / len(fit_mse_array))

print('r2高于0.98的样本数：', r2_num)
print('r2高于0.98的比例：', r2_num / len(fit_mse_array))

plt.figure()
plt.scatter(np.arange(len(fit_mse_array)), fit_mse_array, s=0.5, alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('MSE')
plt.title('the MSE of different samples')
plt.savefig(director_name + f'the MSE of samples.png')

plt.figure()
plt.scatter(np.arange(len(r2_array)), r2_array, s=0.5, alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Goodness of Fit: R$^2$')
plt.title('the R$^2$ of different samples')
plt.savefig(director_name + f'the R$^2$ of samples.png')

plt.show()
