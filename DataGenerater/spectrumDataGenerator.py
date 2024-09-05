import pickle
import numpy as np
import matplotlib.pyplot as plt


# 注意在建立数据集时，没有被随机选中的点被初始化为1，
# 这需要根据前边定义的洛伦兹函数取负后是否加上了最大值来变更，
# 这可能会影响之后的归一化和训练结果
# 另外，随机取样后未被取样到的点设置为1还是0.1可能会影响最后预测结果

# 无随机噪声的洛伦兹函数
def lorentzian(x, x0, gamma):
    return (1 / np.pi) * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2))


# 添加随机噪声的洛伦兹函数,先关于X对称（取负）再加上最大值（峰值）
# 这里注意，如果不加最大值（1 / (np.pi * gamma)），预测效果将非常好（仅需很少的训练次数）（可能是因为有负数的原因）
# 同时会返回一个list，包含峰中心位置、半峰宽
def noisy_lorentzian(x, x0, gamma, lambda_var, noise_level, ):
    if lambda_var < 1 / 2:
        gamma = gamma * ((lambda_var) ** (1 / 2))
    signal = 1 - (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2 / lambda_var))
    noise = noise_level * lambda_var * np.random.normal(size=len(x))
    return signal + noise * signal, [x0, gamma, lambda_var]


# 添加随机噪声的双峰洛伦兹函数
# 返回一个list，包含峰中心位置、半峰宽
# 前两个元素为中心位置，后两个元素为半峰宽
def noisy_lorentzian_2(x, delta_x0, gamma, lambda_var, D, noise_level):
    peak1, label1 = noisy_lorentzian(x, delta_x0, gamma, lambda_var, noise_level)
    peak2, label2 = noisy_lorentzian(x, 2 * D - delta_x0, gamma, lambda_var, noise_level)
    signal = peak1 + peak2
    return signal, [label1[0], label2[0], label1[1], label2[1], label1[2], label2[2]]


# 添加随机噪声的四峰洛伦兹函数
# 返回一个list，包含峰中心位置、半峰宽
# 前四个元素为中心位置，后四个元素为半峰宽
def noisy_lorentzian_4(x, delta_x0, gamma, lambda_var, D, Hyperfine_interaction, noise_level):
    signal1, label1 = noisy_lorentzian_2(x, delta_x0, gamma, lambda_var, delta_x0 - Hyperfine_interaction, noise_level)
    signal2, label2 = noisy_lorentzian_2(x, 2 * D - delta_x0, gamma, lambda_var,
                                         2 * D - delta_x0 + Hyperfine_interaction,
                                         noise_level)
    signal = signal1 + signal2
    return signal, [label1[0], label1[1], label2[0], label2[1], label1[2], label1[3], label2[2], label2[3], label1[4],
                    label1[5], label2[4], label2[5]]


# 定义一些参数  (通过改变data_type的值来调整生成数据的类型)
dir_path = 'lorentzianSpectrum_202_contrast/'
dir_name = ['SinglePeak/', 'DoublePeak/', 'FourPeak/', 'classifier/', 'testData/SinglePeak/', 'testData/DoublePeak/',
            'testData/FourPeak/', 'testData/classifier/', 'validateData/']
data_type = 6  # 0~3为训练用的数据，4~7为测试用的数据，8为最终测试用的数据
dir_name_type = dir_path + dir_name[data_type]  # 保存文件的目录

spectrum_num = 0.00054  # 这是峰中心位置的间隔，直接影响生成的谱线数。频率范围在（2.77, 2.98）之间
spectrum_sampling = 202  # 谱线的采样数，就是每个谱线的频点数
D = 2.87  # 这是零场劈裂常数，对应双峰或多峰分裂时的中心位置
Hyperfine_interaction = 0.02  # 这是超精细相互作用带来的劈裂（也可以是类似的作用）,对应四峰分裂时
noisy_level = 0.008  # 这是模拟生成谱线数据的噪声等级,双峰时噪声会叠加为2倍，四峰4倍
gamma = 0.009  # 这是模拟生成洛伦兹谱线时的半峰宽
lambda_list_train = [1, 1 / 2, 3 / 4., 2 / 3., 2. / 5., 1. / 10., 1. / 15., 1 / 20, 1. / 40.]  # 该参数调整生成谱线对比度(用于训练数据)
lambda_list_test = [4 / 5, 5 / 7, 1 / 3, 1. / 25.]  # 用于测试数据
lambda_list_lastTest = [3 / 5., 4 / 7, 3 / 10, 1 / 18]  # 用于最终测试数据
x_range = np.linspace(2.62, 3.12, spectrum_sampling)  # 频率范围（GHz）

# 生成数据集，先生成谱线数据
center_positions = np.arange(2.72, 2.82, spectrum_num)
spectrum_data = []  # 存放谱线数据
label_fitParams_data = []  # 存放label用于拟合的参数数据
label_classifier = []  # 存放对分类器进行训练的参数
spectrum_data_label = []  # 合并存放谱线数据以及参数数据

# 根据选择的数据类型设定lambda_list
if (data_type >= 0 and data_type <= 3):
    lambda_list = lambda_list_train
elif (data_type >= 4 and data_type <= 7):
    lambda_list = lambda_list_test
elif (data_type == 8):
    lambda_list = lambda_list_lastTest
else:
    assert False  # 不存在上述情况时，强制停止

# 对每个中心位置生成单峰、双峰、四峰谱线
# 注意label_data中label没有*100
for lambda_var in lambda_list:
    # 生成谱线数据时，只保留需要生成的谱线，其他谱线注释掉
    # 生成分类器的数据时，将参数注释掉，保留各谱线（见代码）
    # 修改后注意更改保存目录
    for center_position in center_positions:

        if data_type == 0 or data_type == 4:
            # 单峰数据
            y_values, label_fitParam = noisy_lorentzian(x_range, center_position, gamma, lambda_var, noisy_level)
            spectrum_data.append(y_values)
            label_classifier.append([1, 0, 0, 0])
            label_fitParams_data.append(label_fitParam)
            spectrum_data_label.append(label_fitParam + y_values.tolist())  # 注意label包含3个元素
        elif data_type == 1 or data_type == 5:
            # 双峰谱线数据
            y_values, label_fitParam = noisy_lorentzian_2(x_range, center_position, gamma, lambda_var, D, noisy_level)
            spectrum_data.append(y_values)
            label_classifier.append([0, 1, 0, 0])
            label_fitParams_data.append(label_fitParam)
            spectrum_data_label.append(label_fitParam + y_values.tolist())  # 注意label包含6个元素
        elif data_type == 2 or data_type == 6:
            # 四峰谱线数据
            y_values, label_fitParam = noisy_lorentzian_4(x_range, center_position, gamma, lambda_var, D,
                                                          Hyperfine_interaction, noisy_level)
            spectrum_data.append(y_values)
            label_classifier.append([0, 0, 0, 1])
            label_fitParams_data.append(label_fitParam)
            spectrum_data_label.append(label_fitParam + y_values.tolist())  # 注意label包含12个元素
        elif data_type == 3 or data_type == 7:  #分类器训练和测试数据
            # 单峰谱线数据
            y_values, label_fitParam = noisy_lorentzian(x_range, center_position, gamma, lambda_var, noisy_level)
            spectrum_data.append(y_values)
            label_classifier.append([0, 1, 0, 0])
            # 双峰谱线数据
            y_values, label_fitParam = noisy_lorentzian_2(x_range, center_position, gamma, lambda_var, D, noisy_level)
            spectrum_data.append(y_values)
            label_classifier.append([0, 1, 0, 0])
            # 四峰谱线数据
            y_values, label_fitParam = noisy_lorentzian_4(x_range, center_position, gamma, lambda_var, D,
                                                          Hyperfine_interaction, noisy_level)
            spectrum_data.append(y_values)
            label_classifier.append([0, 0, 0, 1])
        elif data_type == 8:  # 最终测试数据
            # 单峰数据
            y_values, label_fitParam = noisy_lorentzian(x_range, center_position, gamma, lambda_var, noisy_level)
            spectrum_data.append(y_values)
            label_classifier.append([1, 0, 0, 0])
            label_fitParams_data.append(label_fitParam)
            # 双峰谱线数据
            y_values, label_fitParam = noisy_lorentzian_2(x_range, center_position, gamma, lambda_var, D, noisy_level)
            spectrum_data.append(y_values)
            label_classifier.append([0, 1, 0, 0])
            label_fitParams_data.append(label_fitParam)
            # 四峰谱线数据
            y_values, label_fitParam = noisy_lorentzian_4(x_range, center_position, gamma, lambda_var, D,
                                                          Hyperfine_interaction, noisy_level)
            spectrum_data.append(y_values)
            label_classifier.append([0, 0, 0, 1])
            label_fitParams_data.append(label_fitParam)
        else:
            assert False  # data_type不在上述范围时，强制停止
spectrum_data = np.array(spectrum_data)  # 现在我只需要谱线作为训练数据
spectrum_data_label = np.array(spectrum_data_label)  # 注意spectrum_data_label的label数据乘了100
# 打印数据集的形状
print("spectrum Data Shape:", spectrum_data.shape)
print("spectrum Data Label Shape:", spectrum_data_label.shape)
# 归一化
spectrum_data_nomalized = spectrum_data / spectrum_data.max(axis=1, keepdims=True)
# 画出谱线数据
num = 40
sample_indices = []
for i in range(num):
    sample_indices.append(int(i * len(spectrum_data_nomalized) / (num + 1)))
# 画出num张图
for index in sample_indices:
    plt.figure(figsize=(12, 6))

    # 真实谱线
    plt.plot(x_range, spectrum_data_nomalized[index],
             label='True Output Spectrum', linestyle='--', alpha=0.7)
    # 谱线峰分类
    plt.scatter(x_range[0], np.array(spectrum_data_nomalized[index]).min(),
                label='PeakNum' + str(label_classifier[index]))

    plt.title(f'Spectrum Comparison (Sample {index + 1})')
    plt.xlabel('Frequency(GHz)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig(dir_name_type + f'figure_sample{index + 1}.png')
#plt.show()

# -------------此处用于对比谱线线性变换前后是否相同----------------
# 这段代码对原代码无任何影响，仅当生成单峰、双峰、四峰训练数据时可运行（data_type=0,1,2,4,5,6）
'''
spectrum_lambda_var_1 = spectrum_data_nomalized[0]  # lambda_var=1的谱线
spectrum_lambda_var_1_45 = spectrum_data_nomalized[2 * len(spectrum_data_nomalized) // 3 + 1]  # lambda_var=的谱线

# 再次归一化
spectrum_lambda_var_1_normalized = (spectrum_lambda_var_1 - spectrum_lambda_var_1.min()) / (
        spectrum_lambda_var_1.max() - spectrum_lambda_var_1.min())
spectrum_lambda_var_1_45_normalized = (spectrum_lambda_var_1_45 - spectrum_lambda_var_1_45.min()) / (
        spectrum_lambda_var_1_45.max() - spectrum_lambda_var_1_45.min())
# 画图对比
plt.figure(2)
plt.plot(x_range, spectrum_lambda_var_1_normalized, label='label=' + str(label_fitParams_data[0]))
plt.plot(x_range, spectrum_lambda_var_1_45_normalized,
         label='label=' + str(label_fitParams_data[2 * len(spectrum_data_nomalized) // 3 + 1]))
plt.title("不同的归一化方式")
plt.legend()
plt.show()
'''
# 结果为不同，至少神经网络可以分辨，由此可以解决对比度过低时无法分辨的问题
# -------------此处用于对比谱线线性变换前后是否相同----------------
# 转换为列表
spectrum_data_nomalized = spectrum_data_nomalized.tolist()
spectrum_data_label = spectrum_data_label.tolist()
x_range = x_range.tolist()
# 将spectrum_data_label的谱点数据转换为归一化后的数据

# 保存数据
with open(dir_name_type + 'spectrum_data_normalized.pickle', 'wb') as spectrum_data_nomalized_file:
    pickle.dump(spectrum_data_nomalized, spectrum_data_nomalized_file)
with open(dir_name_type + 'label_classifier.pickle', 'wb') as label_classifier_file:
    pickle.dump(label_classifier, label_classifier_file)
with open(dir_name_type + 'label_data.pickle', 'wb') as label_data_file:
    pickle.dump(label_fitParams_data, label_data_file)
with open(dir_name_type + 'spectrum_data_normalized_label.pickle', 'wb') as spectrum_data_nomalized_label_file:
    pickle.dump(spectrum_data_label, spectrum_data_nomalized_label_file)
with open(dir_name_type + 'x_range.pickle', 'wb') as x_range_file:
    pickle.dump(x_range, x_range_file)
