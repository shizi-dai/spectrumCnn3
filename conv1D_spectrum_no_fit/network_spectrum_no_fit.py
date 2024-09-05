import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.optimize import curve_fit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, LeakyReLU, Concatenate, Reshape, ELU
from tensorflow.keras.layers import Layer


# 定义用于拟合的洛伦兹函数（与之前的区别在于增加了幅度变量A）
# 定义洛伦兹函数
def lorentzian(x, x0, gamma, lambda_var, A):
    A = 1. / 2.
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
def uniformSampling(spectrum_data_nomalized, normalized_params, sample_interval=[3, 4]):
    """
    对输入数据进行均匀取样，每隔 sample_interval 个数据取一次样，未取样到的点设为1。

    参数：
    spectrum_data_nomalized: 输入的二维数组
    normalized_params: 归一化参数的数组
    sample_interval: 取样间隔，默认为3、4

    返回：
    input_data_array: 输入数据的NumPy数组
    output_data_array: 输出数据的NumPy数组
    normalized_params_array: 归一化参数的NumPy数组
    """
    normalized_params = np.array(normalized_params)
    # 初始化空的输入数据和输出数据列表
    input_data_list = []
    output_data_list = []
    normalized_params_array = np.zeros([2, 2, 1])
    # 对每个取样间隔进行操作
    for interval in sample_interval:
        # 对每个一维列表执行操作
        for i in range(len(spectrum_data_nomalized)):
            spectrum = spectrum_data_nomalized[i]
            spectrum_Out = spectrum_data_nomalized[i]
            # 复制spectrum以保留原始数据
            input_data = np.copy(spectrum)

            # 对每个位置进行操作
            for j in range(len(spectrum)):
                # 如果位置不是取样点，则设为1
                if j % interval != 0:
                    input_data[j] = 1

            # 合并数据到列表中
            input_data_list.append(input_data)
            output_data_list.append(spectrum_Out)
        normalized_params_array = np.concatenate((normalized_params_array, normalized_params), axis=1)
    # 将列表转换为NumPy数组
    input_data_array = np.array(input_data_list)
    output_data_array = np.array(output_data_list)
    # 对于normalized_param_array,删除初始化时的前两个数
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


# 一些重要参数
peakNum_director_name_array = ['SinglePeak/','DoublePeak/','','FourPeak/']
peakNum = 4 # 使用这个参数控制峰类型（1：单峰，2：双峰，4：四峰）
peakNum_director_name = peakNum_director_name_array[peakNum-1]
# -------------------------准备数据---------------------------------
# 导入已经生成好的数据，类型为list
train_data_dir_name = '../DataGenerater/lorentzianSpectrum_202_contrast/' + peakNum_director_name
with open(train_data_dir_name + 'spectrum_data_normalized.pickle', 'rb') as spectrum_data_normalized_file:
    spectrum_data = pickle.load(spectrum_data_normalized_file)
with open(train_data_dir_name + 'x_range.pickle', 'rb') as x_range_file:
    x_range = pickle.load(x_range_file)
# 将数据从list转换为numpy的array
spectrum_data = np.array(spectrum_data)
x_range = np.array(x_range)

# 画出谱线数据
plt.figure(1)
plt.plot(x_range, spectrum_data[2])
# plt.show()
# 这里我们假设提前知道了谱线的最大最小值，因为对于同一个实验设备，谱线的对比度一般可以提前预知，并基本不变
# -------------------对谱线数据进行归一化编码--------------------------
spectrum_data_nomalized, normalized_params = encoder(spectrum_data)
print('spectrum_data_nomalized', spectrum_data_nomalized.shape)
print('normalized_params', np.array(normalized_params).shape)
# -----------------------------------------------------------------

# 前面已经导入了谱线
# 下面建立数据集
# 将减少采样的谱线作为输入，原始谱线作为输出
# 均匀采样，同时返回归一化参数
input_data_array, output_data_array, normalized_params_array = uniformSampling(spectrum_data_nomalized,
                                                                               normalized_params)
print('normalized_params_array', np.array(normalized_params_array).shape)

# 转换数据为 TensorFlow 张量
input_data_array=input_data_array.reshape((input_data_array.shape[0],input_data_array.shape[1],1))
output_data_array=output_data_array.reshape((output_data_array.shape[0],output_data_array.shape[1],1))

input_data_tensor = tf.constant(input_data_array, dtype=tf.float32)
output_data_tensor = tf.constant(output_data_array, dtype=tf.float32)

Dataset = tf.data.Dataset.from_tensor_slices((input_data_tensor, output_data_tensor))
# 重复数据集一次
dataset = Dataset.repeat(count=2)

# 设置一个足够大的缓冲区来打乱整个数据集
buffer_size = 2 * len(input_data_array)
# 打乱数据集
dataset = dataset.shuffle(buffer_size=buffer_size)

# 分割数据集为训练集和验证集
# 假设我们用 80% 的数据作为训练，20% 的数据作为验证
train_size = int(0.8 * len(input_data_array))
validate_size = len(input_data_array) - train_size

# 现在分割数据集
train_dataset = dataset.take(train_size).batch(1000)
validate_dataset = dataset.skip(train_size).take(validate_size).batch(1000)

# -------------------------建立模型-------------------------------------------------
# 定义一个同时输出谱点，峰个数，中心位置，半峰宽的神经网络
# 定义输入层，形状为[None, 202]
input_tensor = Input(shape=(len(spectrum_data_nomalized[0]), 1))
# 定义第一个卷积全连接网络
dense_layer_1_input = Conv1D(filters=16, kernel_size=20, activation='relu', )(input_tensor)
# dense_layer_1_input = Conv1D(filters=16,kernel_size=15,activation='relu',)(dense_layer_1_input)
dense_layer_1_2 = Flatten()(dense_layer_1_input)
dense_layer_1_3 = Dense(256, activation='linear')(dense_layer_1_2)
dense_layer_1_4 = LeakyReLU(0.18)(dense_layer_1_3)
dense_layer_1_5 = Dense(404, activation='linear')(dense_layer_1_4)
dense_layer_1_6 = LeakyReLU(0.18)(dense_layer_1_5)

# 谱点的数量，注意，这里假设谱点数量为偶数
out_shape_3and4 = spectrum_data_nomalized.shape[1]
# 定义第三个神经网络，接收第一个网络的信息，用于输出前半谱数据
dense_layer_1_6_reshape = Reshape((dense_layer_1_6.shape[1], 1))(dense_layer_1_6)
dense_layer_3_3 = Dense(202, activation='linear')(dense_layer_1_6)
dense_layer_3_4 = LeakyReLU(0.18)(dense_layer_3_3)
dense_layer_3_5 = Dense(out_shape_3and4 // 2, activation='linear')(dense_layer_3_4)
dense_layer_3_6 = LeakyReLU(0.18)(dense_layer_3_5)
# 建立第四个神经网络，接收第一个神经网络的输入，返回后半谱的信息
dense_layer_1_6_reshape = Reshape((dense_layer_1_6.shape[1], 1))(dense_layer_1_6)
dense_layer_4_3 = Dense(202, activation='linear')(dense_layer_1_6)
dense_layer_4_4 = LeakyReLU(0.18)(dense_layer_4_3)
dense_layer_4_5 = Dense(out_shape_3and4 // 2, activation='linear')(dense_layer_4_4)
dense_layer_4_6 = LeakyReLU(0.18)(dense_layer_4_5)
# 将第3个神经网络、第4个神经网络的输出层合并，作为总的输出层

dense_layer_output = Concatenate()([dense_layer_3_6, dense_layer_4_6])

# 定义输出层
# output_tensor = Dense(210,activation='linear')(dense_layer_output)
output_tensor = dense_layer_output
# 创建模型
model = Model(inputs=input_tensor, outputs=output_tensor)

# 打印模型概要
model.summary()

# -----------------------编译训练--------------------------------------------------
# 编译模型
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='mse')
# 定义 ModelCheckpoint 回调
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='Model_contrast/' + peakNum_director_name + 'model_spectrum.keras',  # 保存的模型文件路径
    monitor='loss',  # 监视的指标，可以是 'val_loss' 或其他指标
    save_best_only=True,  # 只保存在验证集上性能最好的模型
    save_weights_only=False,  # 只保存权重
    mode='min',  # 模型保存的依据是最小化损失函数值
    verbose=1  # 显示保存模型的信息
)
# 训练模型
history = model.fit(
    train_dataset, validation_data=validate_dataset,
    epochs=100, batch_size=10000, verbose=1,
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
plt.savefig('Model_contrast/' + peakNum_director_name + 'Training_and_Validation_Loss.png')
# -----------------------载入模型------------------------------

model = tf.keras.models.load_model('Model_contrast/' + peakNum_director_name + 'model_spectrum.keras')

# -----------------------测试模型--------------------------------
# 准备测试数据
test_data_dir_name = '../DataGenerater/lorentzianSpectrum_202_contrast/testData/' + peakNum_director_name
with open(test_data_dir_name + 'spectrum_data_normalized.pickle', 'rb') as test_data_file:
    test_spectrum_data = pickle.load(test_data_file)
    test_spectrum_data = np.array(test_spectrum_data)
with open(test_data_dir_name + '/x_range.pickle', 'rb') as x_range_flie:
    x_range = pickle.load(x_range_flie)
    x_range = np.array(x_range)

# --------------对谱线进行编码------------------------------------------------------
test_spectrum_data_normalized, test_normalized_params = encoder(test_spectrum_data)
# 均匀采样，同时返回归一化参数
test_input_data_array, test_output_data_array, test_normalized_params_array = uniformSampling(
    test_spectrum_data_normalized, test_normalized_params)

# 使用模型对测试数据进行预测
model_predict_data = model.predict(test_input_data_array)

# 对预测谱线进行解码
model_predict_data = decoder(model_predict_data, test_normalized_params_array)
test_output_data_array = decoder(test_output_data_array, test_normalized_params_array)
test_input_data_array = decoder(test_input_data_array, test_normalized_params_array)
# -------------------------结果对比-----------------------------------
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

    # 真实输出谱线
    plt.plot(x_range, test_output_data_array[index],
             label='True Output Spectrum', linestyle=':', alpha=1)

    # 模型预测输出谱线
    plt.plot(x_range, model_predict_data[index],
             label='Model Predicted Spectrum', linestyle='--')

    plt.title(f'Spectrum Comparison (Sample {index + 1})')
    plt.xlabel('frequency(GHz)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig('Model_contrast/' + peakNum_director_name + f'figure_sample{index + 1}.png')
plt.show()
