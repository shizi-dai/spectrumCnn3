import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.optimize import curve_fit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, LeakyReLU, Concatenate, Reshape, ELU
from tensorflow.keras.layers import Layer, Dropout


# 定义用于拟合的洛伦兹函数（与之前的区别在于增加了幅度变量A）
# 定义洛伦兹函数
def lorentzian(x, x0, gamma, lambda_var, A):
    return A - A / (1. / lambda_var + ((x - x0) / gamma) ** 2)


# 定义包含两个洛伦兹函数的函数，双峰函数
# 注意，使用下面的函数进行预测时，传入参数需加‘*’
def lorentzian2(x, *params):
    x0 = params[0:2]
    gamma = params[2:4]
    lambda_var = params[4:6]
    A = params[6:8]

    y = 0
    for i in range(2):
        y += lorentzian(x, x0[i], gamma[i], lambda_var[i], A[0])

    return y


# 定义包含四个洛伦兹函数的函数
# 注意，使用下面的函数进行预测时，传入参数需加‘*’
def lorentzian4(x, *params):
    x0 = params[0:4]
    gamma = params[4:8]
    lambda_var = params[8:12]
    A = params[12:16]

    y = 0
    for i in range(4):
        y += lorentzian(x, x0[i], gamma[i], lambda_var[i], A[0])

    return y


# 定义随机取样函数
def randSampling(spectrum_data_normalized, spectrum_data_normalized_label, randSampleNum=[31, 51, 71], repeats=20):
    # spectrum_data_nomalized为np.array
    # 返回input_data_array、output_data_array
    # 未被取样到的点设为1

    # 初始化空的输入数据和输出数据列表
    input_data_list = np.array([spectrum_data_normalized[0]])
    output_data_list = np.array([spectrum_data_normalized_label[0]])

    # 重复执行随机采样操作并合并数据
    for x in randSampleNum:
        for _ in range(repeats):
            # 复制spectrum以保留原始数据
            input_data = np.copy(spectrum_data_normalized)

            # 对每个一维列表执行操作
            for i in range(spectrum_data_normalized.shape[0]):
                # 随机选择 x 个位置
                selected_positions = np.random.choice(spectrum_data_normalized.shape[1], x, replace=False)

                # 将选定位置的值保留，其余位置的值设为1
                input_data[i, :] = 1
                input_data[i, selected_positions] = spectrum_data_normalized[i, selected_positions]

            # 合并数据到列表中
            input_data_list = np.concatenate((input_data_list, input_data), axis=0)
            output_data_list = np.concatenate((output_data_list, spectrum_data_normalized_label), axis=0)

    # 将列表转换为NumPy数组
    input_data_array = np.array(input_data_list)
    output_data_array = np.array(output_data_list)

    # 打印结果
    print("Input Data Shape:", input_data_array.shape)
    print("Output Data Shape:", output_data_array.shape)

    return input_data_array, output_data_array


# 定义均匀取样函数
def uniformSampling(spectrum_data_nomalized, label_data, normalizedParams, sample_interval=[1]):
    """
    对输入数据进行均匀取样，每隔 sample_interval 个数据取一次样，未取样到的点设为1。

    参数：
    spectrum_data_nomalized: 输入的二维数组
    label_data: 带标记的二维数组，作为输出数据
    sample_interval: 取样间隔，默认为2

    返回：
    input_data_array: 输入数据的NumPy数组
    output_data_array: 输出数据的NumPy数组
    """
    normalizedParams = np.array(normalizedParams)
    # 初始化空的输入数据和输出数据列表
    input_data_list = []
    output_data_list = []
    normalized_params_array = np.zeros([2, 2, 1])
    # 对每个取样间隔进行操作
    for interval in sample_interval:
        # 对每个一维列表执行操作
        for i in range(len(spectrum_data_nomalized)):
            spectrum = spectrum_data_nomalized[i]
            spectrum_label = label_data[i]
            # 复制spectrum以保留原始数据
            input_data = np.copy(spectrum)

            # 对每个位置进行操作
            for j in range(len(spectrum)):
                # 如果位置不是取样点，则设为1
                if j % interval != 0:
                    input_data[j] = 1

            # 合并数据到列表中
            input_data_list.append(input_data)
            output_data_list.append(spectrum_label)
        normalized_params_array = np.concatenate((normalized_params_array, normalizedParams), axis=0)
    # 将列表转换为NumPy数组
    input_data_array = np.array(input_data_list)
    output_data_array = np.array(output_data_list)
    # 对于normalized_param_array,删除初始化时的前两个一维列表
    normalized_params_array = np.array(normalized_params_array[:, 2:, :])
    # 打印结果
    print("Input Data Shape:", input_data_array.shape)
    print("Output Data Shape:", output_data_array.shape)

    return input_data_array, output_data_array, normalized_params_array


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


# 自定义损失函数,效果奇差
def custom_loss_doublePeak(y_true, y_pred):
    # 对于双峰，前面的6个参数返回的是2中心位置、2半峰宽、2对比调制参数
    loss1 = tf.reduce_mean(tf.square((y_true[0:2] - y_pred[0:2]) * 100))  # 中心位置损失
    loss2 = tf.reduce_mean(tf.square((y_true[2:4] - y_pred[2:4]) * 100))  # 半峰宽损失
    loss3 = tf.reduce_mean(tf.square((y_true[4:6] - y_pred[4:6]) * 100))  # 调制参数损失
    loss4 = tf.reduce_mean(tf.square((y_true[6:y_true.shape[0]] - y_pred[6:y_pred.shape[0]]) * 50))  # 谱点损失
    loss = loss1 + loss2 + loss3 + loss4  # 总损失
    return loss


# 自定义激活函数,效果一般
class PReLU(Layer):
    def __init__(self, alpha_initializer='zeros', **kwargs):
        super(PReLU, self).__init__(**kwargs)
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha',
                                     shape=(1,),
                                     initializer=self.alpha_initializer,
                                     trainable=True)

    def call(self, inputs):
        return tf.maximum(0.0, inputs) + self.alpha * tf.minimum(0.0, inputs)


class RandomizedLeakyReLU(Layer):
    def __init__(self, alpha_range=(0.001, 0.1), **kwargs):
        super(RandomizedLeakyReLU, self).__init__(**kwargs)
        self.alpha_range = alpha_range

    def build(self, input_shape):
        self.alpha = self.add_weight("alpha", initializer="zeros", trainable=True)

    def call(self, inputs):
        random_alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1], size=10)
        return tf.nn.leaky_relu(inputs, alpha=float(random_alpha[0]))


# 一些重要参数
peakNum_director_name_list = ['SinglePeak/', 'DoublePeak/', '', 'FourPeak/']
label_factor = [1, 20, 1]  # 对label放缩的基础系数

peakNum = 4  # 使用这个参数控制峰类型（1：单峰，2：双峰，4：四峰）
peakNum_director_name = peakNum_director_name_list[peakNum - 1]
label_factor_expand = np.repeat(label_factor, peakNum)  # 对label放缩的系数，根据峰类型调整长度

# 导入已经生成好的数据，类型为list
train_data_dir_name = '../../DataGenerater/lorentzianSpectrum_202_contrast/' + peakNum_director_name
with open(train_data_dir_name + 'spectrum_data_normalized.pickle', 'rb') as spectrum_data_file:
    spectrum_data = pickle.load(spectrum_data_file)
with open(train_data_dir_name + 'label_data.pickle', 'rb') as label_data_file:
    label_data = pickle.load(label_data_file)
with open(train_data_dir_name + 'x_range.pickle', 'rb') as x_range_file:
    x_range = pickle.load(x_range_file)
# 将数据从list转换为numpy的array
# spectrum_data_normalized_label与spectrum_data_normalized是二维数组
# 但比后者，前者在每个一维数组的前面多出来几个元素，作为label
# 注意，这里对他们乘以label_factor_expand，以增大损失函数
spectrum_data = np.array(spectrum_data)
label_data = np.array(label_data) * label_factor_expand  # 对label进行放缩
x_range = np.array(x_range)

# 画出谱线数据
plt.figure(1)
plt.plot(x_range, spectrum_data[2])
# plt.show()

# 前面已经导入了谱线
# 下面建立数据集
# 将减少采样的谱线作为输入，原始谱线作为输出

# 编码归一化
# normalizedParams = np.ones([2, 2, 1])
# spectrum_data_normalized, normalizedParams = encoder(spectrum_data)

# 均匀采样
# input_data_array, output_data_array, normalizedParams_array = uniformSampling(spectrum_data, label_data,
#                                                                              normalizedParams)
input_data_array = spectrum_data.reshape((spectrum_data.shape[0], spectrum_data.shape[1], 1))
output_data_array = np.array(label_data)[:, 1 * peakNum:3 * peakNum]

# 转换数据为 TensorFlow 张量
input_data_tensor = tf.constant(input_data_array, dtype=tf.float32)
output_data_tensor = tf.constant(output_data_array, dtype=tf.float32)

Dataset = tf.data.Dataset.from_tensor_slices((input_data_tensor, output_data_tensor))
# 重复数据集一次
dataset = Dataset.repeat(count=2)

# 设置一个足够大的缓冲区来打乱整个数据集
buffer_size = len(2*input_data_array)
# 打乱数据集
dataset = dataset.shuffle(buffer_size=buffer_size)

# 分割数据集为训练集和验证集
# 假设我们用 80% 的数据作为训练，20% 的数据作为验证
train_size = int(0.8 * len(input_data_array))
validate_size = len(input_data_array) - train_size

# 现在分割数据集
train_dataset = dataset.take(train_size).batch(1000)
validate_dataset = dataset.skip(train_size).take(validate_size).batch(1000)

# 定义简单的一维卷积神经网络,本文件未使用该模型
model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=16, kernel_size=10, padding='causal', dilation_rate=2, activation='relu',
                           input_shape=(len(input_data_array[0]), 1)),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Flatten(input_shape=(len(input_data_array[0]),)),  # 输入层
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.LeakyReLU(alpha=0.05),
    tf.keras.layers.Dense(64, activation='linear'),
    tf.keras.layers.LeakyReLU(alpha=0.05),
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.LeakyReLU(alpha=0.05),
    tf.keras.layers.Dense(len(output_data_array[0]))
])

# --------------------------------------------------------------------------
# 定义一个输出半峰宽、对比调制参数的神经网络
# 定义输入层，形状为[None, 202]
input_tensor = Input(shape=(len(input_data_array[0]), 1))
# 定义第一部分卷积全连接网络
dense_layer_1_input = Conv1D(filters=8, kernel_size=8, activation='relu', )(input_tensor)
dense_layer_1_input = Conv1D(filters=16, kernel_size=10, activation='relu', )(dense_layer_1_input)
dense_layer_1_input = Conv1D(filters=16, kernel_size=12, activation='relu', )(dense_layer_1_input)
dense_layer_1_2 = Flatten()(dense_layer_1_input)
dense_layer_1_3 = Dense(256, activation='linear')(dense_layer_1_2)
dense_layer_1_3 = Dropout(0.5)(dense_layer_1_3)
dense_layer_1_4 = LeakyReLU(0.18)(dense_layer_1_3)
dense_layer_1_5 = Dense(128, activation='linear')(dense_layer_1_4)
dense_layer_1_6 = ELU(1)(dense_layer_1_5)
# 定义第二部分神经网络，接收第一个网络的信息，它用于输出半峰宽、对比调制参数
out_shape_2_4 = output_data_array.shape[1]

dense_layer_2_1 = Dense(64, activation='linear')(dense_layer_1_6)
dense_layer_2_2 = LeakyReLU(0.18)(dense_layer_2_1)
dense_layer_2_3 = Dense(out_shape_2_4, activation='linear')(dense_layer_2_2)
dense_layer_2_4 = ELU(1)(dense_layer_2_3)

# 定义输出层
# output_tensor = Dense(210,activation='linear')(dense_layer_output)
output_tensor = dense_layer_2_4
# 创建模型
model = Model(inputs=input_tensor, outputs=output_tensor)

# 打印模型概要
model.summary()

# ---------------------------------------------------------------------------
# 编译模型
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='mse')
# 定义 ModelCheckpoint 回调
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='Model_contrast/' + peakNum_director_name + 'model_fitParams_halPeWid_contrast.keras',  # 保存的模型文件路径
    monitor='loss',  # 监视的指标，可以是 'val_loss' 或其他指标
    save_best_only=True,  # 只保存在验证集上性能最好的模型
    save_weights_only=False,  # 只保存权重
    mode='min',  # 模型保存的依据是最小化损失函数值
    verbose=1  # 显示保存模型的信息
)
# ----------------------------训练模型
history = model.fit(
    train_dataset, validation_data=validate_dataset,
    epochs=2500, batch_size=1000, verbose=1,
    callbacks=[model_checkpoint]
)

# -----------------------画出损失函数变化------------------------
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Model_contrast/' + peakNum_director_name + 'Training_and_Validation_Loss_halfPeakWitd_contrast.png')
# -------------------------------------------------------------------

# load模型
model = tf.keras.models.load_model(
    'Model_contrast/' + peakNum_director_name + 'model_fitParams_halPeWid_contrast.keras')

# 存放label_factor_expand
with open('Model_contrast/' + peakNum_director_name + 'label_factor_expand.pickle', 'wb') as label_factor_expand_file:
    pickle.dump(label_factor_expand.tolist(), label_factor_expand_file)

# -----------------------------测试模型----------------------------------------------

# 准备测试数据
test_data_dir_name = '../../DataGenerater/lorentzianSpectrum_202_contrast/testData/' + peakNum_director_name
with open(test_data_dir_name + 'spectrum_data_normalized.pickle', 'rb') as test_spectrum_data_file:
    test_spectrum_data = pickle.load(test_spectrum_data_file)
    test_spectrum_data = np.array(test_spectrum_data)
with open(test_data_dir_name + 'label_data.pickle', 'rb') as test_label_data_file:
    test_label_data = pickle.load(test_label_data_file)
    test_label_data = np.array(test_label_data)
with open(test_data_dir_name + '/x_range.pickle', 'rb') as x_range_flie:
    x_range = pickle.load(x_range_flie)
    x_range = np.array(x_range)

# 编码归一化
# test_spectrum_data_normalizedParams = np.ones([2, 2, 1])
test_spectrum_data_normalized, test_spectrum_data_normalizedParams = encoder(test_spectrum_data)
# 均匀采样
# test_input_data_array, test_output_data_array, test_spectrum_data_normalizedParams_array = uniformSampling(
#    test_spectrum_data, test_label_data, test_spectrum_data_normalizedParams)
test_input_data_array = test_spectrum_data
test_output_data_array = test_label_data[:, 1 * peakNum:3 * peakNum]

# 使用模型对测试数据进行预测 (半峰宽、对比度)
model_predict_data = model.predict(test_input_data_array)
model_predict_halPeWid_contrast = model_predict_data / label_factor_expand[1 * peakNum:3 * peakNum]  # 反放缩回原数据

# ---------------导入预测峰中心位置的模型，并预测-----------------------------------

model_peakPosition = tf.keras.models.load_model(
    'Model_contrast/' + peakNum_director_name + 'model_fitParams_peakPosition.keras')
model_predict_peakPosition = model_peakPosition(test_spectrum_data_normalized)
# --------------------------------------------------------------------------------------

model_predict_params = np.concatenate((model_predict_peakPosition, model_predict_halPeWid_contrast), axis=1)

# 查看预测的参数直接生成谱线与真实谱线差距
predict_data_list = []
predict_params_list = []
for i in range(test_input_data_array.shape[0]):
    try:
        if peakNum_director_name == 'SinglePeak/':
            fit_params = list(model_predict_params[i]) + [1]
            predict_data = lorentzian(x_range, *fit_params)
        elif peakNum_director_name == 'DoublePeak/':
            fit_params = list(model_predict_params[i]) + [0.5, 0.5]
            predict_data = lorentzian2(x_range, *fit_params)
        elif peakNum_director_name == 'FourPeak/':
            fit_params = list(model_predict_params[i]) + [0.25, 0.25, 0.25, 0.25]
            predict_data = lorentzian4(x_range, *fit_params)
        else:
            print('NameError: ' + peakNum_director_name + 'not exist')
        predict_data_list.append(predict_data)
        predict_params_list.append(fit_params)
        print('fit_params', fit_params)
    except (RuntimeError, ValueError) as e:
        print(i)
        predict_data_list.append(np.ones(len(x_range)))
        predict_params_list.append(np.zeros(len(fit_params)))

# 选择num个样本进行对比
num = 50
sample_indices = []
for i in range(num):
    sample_indices.append(int(i * len(test_output_data_array) / (num + 1)))
# 输出预测结果
for index in sample_indices:
    plt.figure()
    plt.scatter(x_range, test_input_data_array[index], label='input spectrum')
    plt.plot(x_range, predict_data_list[index], label='the spectrum with predict params')

    print(f'sample {index + 1}')
    print('true params:', test_output_data_array[index])
    print('predict params:', model_predict_params[index])

    plt.xlabel('frequency(GHz)')
    plt.ylabel('signal')
    plt.title(f'predict the params of input spectrum({index+1})')
    plt.legend()
    plt.savefig('Model_contrast/' + peakNum_director_name + f'figure_sample{index + 1}_predictParam.png')
    print('\n')
plt.show()
# --------------------------拟合谱线---------------------------------------------------------------------
# 本段使用洛伦兹函数对真实数据(input data)进行拟合
# 将采用神经网络输出的中心位置、半峰宽作为初始猜测值
spectrum_data_nomalized_fit = np.where(test_input_data_array != 1, test_input_data_array, 0)  # 这是输入的谱点，用于拟合
x_range_fit = np.where(test_input_data_array != 1, x_range, 0)
print('spectrum_data_nomalized_fit.shape', spectrum_data_nomalized_fit.shape)
print('x_range_fit.shape', x_range_fit.shape)

# 初始化拟合列表
predict_fit_data_list = []
predict_fit_params_list = []

for i in range(test_input_data_array.shape[0]):
    print(f'第{i}个，总{test_input_data_array.shape[0]}个')

    spectrum_normalized_fit = spectrum_data_nomalized_fit[i]
    spectrum_normalized_fit_nonzero_indices = np.nonzero(spectrum_normalized_fit)
    spectrum_normalized_fit = spectrum_normalized_fit[spectrum_normalized_fit_nonzero_indices]
    # print('spectrum_normalized_fit', spectrum_normalized_fit.shape)

    x_fit = x_range_fit[i]
    x_fit_nonzero_indices = np.nonzero(x_fit)
    x_fit = x_fit[x_fit_nonzero_indices]
    # print('x_fit', x_fit.shape)

    # 参数界定范围
    bounds_1 = ([x_range[0], -0.1, -0.2, 0], [x_range[len(x_range) - 1], 0.1, 1.5, 2])
    bounds_2 = (
        [x_range[0], x_range[0], -0.1, -0.1, -0.2, -0.2, 0., 0, ],
        [x_range[len(x_range) - 1], x_range[len(x_range) - 1],
         0.1, 0.1, 1.5, 1.5, 2, 2, ])  # DoublePeak
    bounds_4 = (
        [x_range[0], x_range[0], x_range[0], x_range[0], -0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.2, 0., 0., 0.,
         0., ],
        [x_range[len(x_range) - 1], x_range[len(x_range) - 1], x_range[len(x_range) - 1], x_range[len(x_range) - 1],
         0.1, 0.1, 0.1, 0.1, 1.5, 1.5, 1.5, 1.5, 2, 2, 2, 2, ])
    # 初始幅值
    initial_A = []
    if peakNum_director_name == 'SinglePeak/':
        bounds = bounds_1
        initial_A = [1]
    elif peakNum_director_name == 'DoublePeak/':
        bounds = bounds_2
        initial_A = [0.5, 0.5]
    elif peakNum_director_name == 'FourPeak/':
        bounds = bounds_4
        initial_A = [0.25, 0.25, 0.25, 0.25]
    else:
        print('NameError: ' + peakNum_director_name + 'not exist')
    # 参数的初始猜测值,依次为中心位置、半峰宽、lambda_var、幅值

    initial_guess = list(model_predict_params[i]) + initial_A
    print('init_guess', initial_guess)

    try:
        if peakNum_director_name == 'SinglePeak/':
            fit_params, covariance = curve_fit(lorentzian, x_fit, spectrum_normalized_fit, p0=initial_guess,
                                               bounds=bounds, maxfev=20000)
            predict_data = lorentzian(x_range, *fit_params)
        elif peakNum_director_name == 'DoublePeak/':
            fit_params, covariance = curve_fit(lorentzian2, x_fit, spectrum_normalized_fit, p0=initial_guess,
                                               bounds=bounds, maxfev=20000)
            predict_data = lorentzian2(x_range, *fit_params)
        elif peakNum_director_name == 'FourPeak/':
            fit_params, covariance = curve_fit(lorentzian4, x_fit, spectrum_normalized_fit, p0=initial_guess,
                                               bounds=bounds, maxfev=20000)
            predict_data = lorentzian4(x_range, *fit_params)
        else:
            print('NameError: ' + peakNum_director_name + 'not exist')
        predict_fit_data_list.append(predict_data)
        predict_fit_params_list.append(fit_params)
        print('fit_params', fit_params)
    except (RuntimeError, ValueError) as e:
        print(i)
        predict_fit_data_list.append(np.ones(len(x_range)))
        predict_fit_params_list.append(np.zeros(len(initial_guess)))
# predict_fit_data_list包含了拟合后得到的曲线数据
# predict_fit_params_list包含了拟合后曲线的参数信息
# -----------------------------结果可视化------------------------------------------------------------------

# 选择num个样本进行对比
num = 30
sample_indices = []
for i in range(num):
    sample_indices.append(int(i * len(test_output_data_array) / (num + 1)))
# 画出num张对比图
for index in sample_indices:
    plt.figure(figsize=(12, 6))

    # 输入谱线
    plt.scatter(x_range, test_input_data_array[index], label='Input Spectrum')

    # 真实谱线
    plt.plot(x_range, test_spectrum_data[index], label='True Output Spectrum', linestyle='--', alpha=0.7)

    # 模型预测输出谱线
    # plt.plot(x_range, model_predict_data[index][out_shape_2_4:out_shape_2_4 + out_shape_3and4],
    #         label='Model Predicted Spectrum', linestyle='--')

    # 洛伦兹函数拟合谱线
    plt.plot(x_range, predict_fit_data_list[index], label='Lorentzian fit Spectrum', linestyle='--')

    # 对比中心位置、半峰宽的预测情况
    print(f'Sample {index + 1}:')
    print('True params', test_output_data_array[index])
    print('Predicted params', model_predict_params[index], '\n')

    plt.title(f'Spectrum Comparison (Sample {index + 1})')
    plt.xlabel('frequency(GHz)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig('Model_contrast/' + peakNum_director_name + f'figure_sample{index + 1}.png')
plt.show()
