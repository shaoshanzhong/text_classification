In order to grade the hazard levels and probabilities of relevant behaviors and events in the urban rail transit industry, the Transformer model is introduced and solved through the SFT (Supervised Fine-Tuning) pre-training mode.

In terms of the urban rail transit industry, currently the pre-training dataset is limited. As the example of my program, the content of the operation specifications in the rail transit industry is used as the pre-training data. The same approach will be applied to the equipment function area and management function area.

An attempt was made using the Bert model introduced in the well-known Huggingface_Toturials-main. However, the result was overfitting. For the rating requirement from 1 to 10, the output of the result was all 5. Considering the efforts of digging Bert Model, the idea of using the Transformer Model of the feed-forward strategy jumped up and by adjusting the scale of network parameters, the mission was completed.  
(It will be very appreciated for guiding how to enhance the Bert model in Huggingface_Toturials-main to solve the small and bad quality dataset.)

The scale of the model is as follows:  
d_model = 2304  
n_heads = 36  
n_layers = 2  
num_classes = 10  
dim_feedforward = 2048  
dropout = 0.1  
N_EPOCHS = 6

The Loss report.
The original training data can’t be out of the human effect, after the Loss is less than 0.5 the result is completely acceptable.  
Epoch 1/6 training completed, average Loss: 1.362  
Epoch 2/6 training completed, average Loss: 1.128  
Epoch 3/6 training completed, average Loss: 0.959  
Epoch 4/6 training completed, average Loss: 0.801  
Epoch 5/6 training completed, average Loss: 0.642  
Epoch 6/6 training completed, average Loss: 0.489  

Dependent libraries:
torch version: 2.6.0  
pandas version: 2.2.3  
transformers version: 4.49.0  
The descriptions of files:  
create-model.py Build and train the model  
input.xlsx Pre-training data set, NO., Text, Label  
Dlg-context-grade-01B.pth Trained Model output  
use-model.py Use the model  
test.xlsx Test Dataset, NO., Text, Label, Info, (human rating)  
test_result.xlsx Output, The result can be adjusted manually by comparing Label and Info.  

This model is also applicable to the general context grading.  
e-mail: shaoshanzhong@hotmail.com

======================================================================

为给城市轨道交通行业的相关行为、事件进行危害等级、危害概率的分级，引入Transformer模型，通过SFT预训练模式，进行解决。
由于行业的特点，预训练数据集有限，目前以轨道交通行业关于运营规范的内容为参考，进行预训练数据的提取。后续设备类、管理类方式相同。
之前用比较出名的 Huggingface_Toturials-main介绍的 Bert模型进行了尝试，结果过拟合，对于1至10级的评级需求，结果输出都为5。这样考虑放弃 Bert模型，
采用前馈模型的Transformer结构，通过调节网络参数规模，来实现最开始提出的目标规划，目前结果完全可以接受。  
（欢迎熟悉 Bert模型细节的大佬给与指导，以搞清楚Huggingface_Toturials-main中的Bert什么地方需要调整）

模型的规模如下：  
d_model = 2304  
n_heads = 36  
n_layers = 2  
num_classes = 10  
dim_feedforward = 2048  
dropout = 0.1  
N_EPOCHS = 6  

训练过程 Loss的情况如下，由于原训练数据也有一定人为偏差，Loss小于0.5后的模型输出结果已经完全可以接受。  
Epoch 1/6 训练完成，平均 Loss: 1.362  
Epoch 2/6 训练完成，平均 Loss: 1.128  
Epoch 3/6 训练完成，平均 Loss: 0.959  
Epoch 4/6 训练完成，平均 Loss: 0.801  
Epoch 5/6 训练完成，平均 Loss: 0.642  
Epoch 6/6 训练完成，平均 Loss: 0.489  

几个依赖库的版本如下：  
torch version: 2.6.0  
pandas version: 2.2.3  
transformers version: 4.49.0  
文件说明如下：  
create-model.py   建立和训练模型  
input.xlsx   预训练数据集合，格式为序号、文本、标注  
Dlg-context-grade-01B.pth   模型输出，后续使用不必要再浪费时间做训练  
use-model.py   使用模型，对拟评测的数据进行评测  
test.xlsx   拟评测的数据集，格式为序号、文本、标注（空）、人为评分  
test_result.xlsx  推理评测结果，格式为序号、文本、标注（空）、人为评分，可以通过比较 Label和Info，对结果进行人为调整  

该模型也适用于通用的对文字内容进行预训练评级分类。  
e-mail: shaoshanzhong@hotmail.com
