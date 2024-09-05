import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, LeakyReLU, Concatenate, Reshape, ELU
from tensorflow.keras.layers import Layer


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


def uniformSamplingClassifier(spectrum_data_nomalized, label_classifier, sample_interval=[4]):
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
        for i in range(0, len(spectrum_data_nomalized)):
            spectrum = spectrum_data_nomalized[i]
            label = label_classifier[i]

            # 复制spectrum以保留原始数据
            input_data = np.copy(spectrum)

            # 对每个位置进行操作
            for i in range(len(spectrum)):
                # 如果位置不是取样点，则设为1
                if i % interval != 0:
                    input_data[i] = 1

            # 合并数据到列表中
            input_data_list.append(input_data)
            output_data_list.append(label)

    # 将列表转换为NumPy数组
    input_data_array = np.array(input_data_list)
    output_data_array = np.array(output_data_list)

    # 打印结果
    print("Input Data Shape:", input_data_array.shape)
    print("Output Data Shape:", output_data_array.shape)

    return input_data_array, output_data_array


# -----------------------导入数据-----------------------------
train_data_dir_name = '../DataGenerater/lorentzianSpectrum_202_contrast/classifier/'
with open(train_data_dir_name + 'spectrum_data_normalized.pickle', 'rb') as spectrum_data_normalized_file:
    spectrum_data_normalized = pickle.load(spectrum_data_normalized_file)
with open(train_data_dir_name + 'label_classifier.pickle', 'rb') as label_classifier_file:
    label_classifier = pickle.load(label_classifier_file)

# 对谱线数据进行编码（归一化）
spectrum_data_normalized, spectrum_data_normalized_params = encoder(spectrum_data_normalized)
# 均匀取样
input_array, output_array = uniformSamplingClassifier(spectrum_data_normalized, label_classifier)

print('input_array', input_array.shape, input_array[0])
print('output_array', output_array.shape, output_array[0])

input_array=input_array.reshape((input_array.shape[0],input_array.shape[1],1))

input_array_tensor = tf.constant(input_array, dtype=tf.float32)
output_array_tensor = tf.constant(output_array, dtype=tf.float32)

#-----------------------dataset------------------------------
Dataset = tf.data.Dataset.from_tensor_slices((input_array_tensor, output_array_tensor))
# 重复数据集一次
dataset = Dataset.repeat(count=2)

# 设置一个足够大的缓冲区来打乱整个数据集
buffer_size = 2 * len(input_array)
# 打乱数据集
dataset = dataset.shuffle(buffer_size=buffer_size)

# 分割数据集为训练集和验证集
# 假设我们用 80% 的数据作为训练，20% 的数据作为验证
train_size = int(0.8 * len(input_array))
validate_size = len(input_array) - train_size

# 现在分割数据集
train_dataset = dataset.take(train_size).batch(2000)
validate_dataset = dataset.skip(train_size).take(validate_size).batch(2000)


# ---------------------------- 创建模型 ---------------------------------------
model = tf.keras.Sequential([
    Conv1D(filters=8, kernel_size=10, activation='relu', input_shape=(len(input_array[0]), 1)),
    Flatten(),
    Dense(64, activation='linear'),
    LeakyReLU(0.18),
    Dense(32, activation='linear'),
    LeakyReLU(0.18),
    Dense(len(output_array[0]), activation='softmax')  # 输出层，使用softmax激活函数，输出1个值
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 定义 ModelCheckpoint 回调
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='Model_contrast/'+ 'classifier.keras',  # 保存的模型文件路径
    monitor='loss',  # 监视的指标，可以是 'val_loss' 或其他指标
    save_best_only=True,  # 只保存在验证集上性能最好的模型
    save_weights_only=False,  # 只保存权重
    mode='min',  # 模型保存的依据是最小化损失函数值
    verbose=1  # 显示保存模型的信息
)
# ----------------------------训练模型

# 训练模型
history = model.fit(
    train_dataset, validation_data=validate_dataset,
    epochs=500, batch_size=1000, verbose=1
)

# -----------------------画出损失函数变化------------------------
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Model_contrast/'+'Training_and_Validation_Loss.png')
#-------------------------------------------------------------------


# 保存模型
director_name = 'Model_contrast/'


# ---------------------- 测试数据 --------------------------------
test_data_dir_name = '../DataGenerater/lorentzianSpectrum_202_contrast/testData/classifier/'
with open(test_data_dir_name + 'spectrum_data_normalized.pickle', 'rb') as test_spectrum_normalized_file:
    test_spectrum_normalized = pickle.load(test_spectrum_normalized_file)
    test_spectrum_normalized = np.array(test_spectrum_normalized)
with open(test_data_dir_name + 'label_classifier.pickle', 'rb') as test_label_classifier_file:
    test_label_classifier = pickle.load(test_label_classifier_file)
    test_label_classifier = np.array(test_label_classifier)
with open(test_data_dir_name + 'x_range.pickle', 'rb') as test_x_range_file:
    test_x_range = pickle.load(test_x_range_file)
    test_x_range = np.array(test_x_range)

print('test_spectrum_normalized', test_spectrum_normalized.shape, test_spectrum_normalized[0])
print('test_label_classifier', test_label_classifier.shape, test_label_classifier[0])
print('test_x_range', test_x_range.shape)

#编码（归一化）
test_spectrum_normalized,test_spectrum_normalized_params=encoder(test_spectrum_normalized)
#均匀取样
test_spectrum_normalized_input, test_label_classifier_output = uniformSamplingClassifier(test_spectrum_normalized,
                                                                                         test_label_classifier)
# ------------------------测试结果---------------------------
model_predict_classifier = model.predict(test_spectrum_normalized_input)

# 选择num个样本进行对比
num = 30
sample_indices = []
for i in range(num):
    sample_indices.append(int(i * len(test_label_classifier_output) / (num + 1)))
# 画出num张对比图
for index in sample_indices:
    plt.figure(figsize=(12, 6))

    # 输入谱线
    plt.scatter(test_x_range, test_spectrum_normalized_input[index], label='Input Spectrum')
    # 真实谱线
    plt.plot(test_x_range, test_spectrum_normalized[index], label='True spectrum')
    # 峰数量的预测情况
    plt.scatter(test_x_range[0], test_spectrum_normalized_input[index].min(),
                label='Model predict' + str(model_predict_classifier[index]))
    plt.scatter(test_x_range[0], 1, label='True output' + str(test_label_classifier_output[index]))
    print('True Output', test_label_classifier_output[index])
    print('Model Predicted', model_predict_classifier[index], '\n')

    plt.title(f'Spectrum  (Sample {index + 1})')
    plt.xlabel('Frequency(GHz)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.savefig(director_name + f'/figure_sample{index + 1}.png')
plt.show()
