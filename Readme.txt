以下是对文件目录的说明

DataGenerater：存放了生成的谱线数据，使用 pickle 模块以python列表形式存放，具体格式见目录内的python文件

Classifier：训练的分类器模型，保存了训练结果和训练模型

conv1D_fit_withFullSpectrum：谱线参数预测模型，包含峰中心位置预测模型和半峰宽、对比度预测模型。分别由文件
			  network_fit_withFullSpectrum_peakPosition.py、network_fit_withFullSpectrum_peakPosition.py构建。

conv1D_spectrum_no_fit：谱线还原模型，保存了欠采样谱线还原模型的训练结果和模型

Unity_Model：对各模型的整合，实现了对混合欠采样谱线的分类、还原、参数预测和拟合。运行unity_Model.py即可，结果存放于results中。


！！注意：目前本文件中没有保存任何数据和结果，只有源代码。若需要运行和训练，请按以下顺序运行文件。！！

1. 运行DataGenerater目录下的spectrumDataGenerator.py文件，将data_type从0至8依次取值并运行，生成谱线数据。

2. 运行Classifier目录下的Classifier.py文件。

3. 运行conv1D_spectrum_no_fit目录下的network_spectrum_no_fit.py，将peakNum取1、2、4，分别运行。

4. 运行conv1D_fit_withFullSpectrum目录下的network_fit_withFullSpectrum_peakPosition.py文件，将peakNum取1、2、4，分别运行；

5. 运行conv1D_fit_withFullSpectrum目录下的network_fit_withFullSpectrum_haifPeWid_contras.py文件，将peakNum取1、2、4，分别运行；

6. 运行Unity_Model目录下的unity_Model.py文件。

！！第1步必须为第一步，第6步必须为最后一步；第4步与第5步顺序不能颠倒。！！

