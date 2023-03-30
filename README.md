# 基于对比学习对MNIST数据集进行预训练和分类
## 什么是对比学习？
对比学习是一种机器学习方法，它利用训练数据中的相似和不同之处来学习分类任务。对比学习的目标是通过将相似样本归为一类、将不同样本归为另一类来学习分类决策边界。这种方法通常用于处理具有少量标注数据的问题，例如人脸识别，图像检索和自然语言处理。
## 模型构架及训练步骤
### 1.加载并预处理数据集
   可以使用本地的数据集，也可以在线下载(需要外网)，建议直接使用本地
```python
# 指定本地路径的方法
from keras.datasets import mnist
path = '你的数据集路径'
(x_train, y_train), (x_test, y_test) = mnist.load_data(path)
# 数据归一化处理
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```
### 2.自定义DataGenerate
   在这个类中锚点数据将从数据集中直接按标签取出，将锚点数据送进数据增强器，进行随机旋转、平移、剪切、缩放、翻转等方法。生成正样本。
   在原有的数据集中选择标签不等于当前标签的数据作为负样本。
### 3.自定义损失函数
   采用欧式距离计算正负样本和锚点之间的差异，并设置alpha参数作为阈值当negative样本与anchor样本的距离大于该阈值时，我们认为这样个样本是不匹配的，需要具体问题具体调整。
### 4.构建预训练网络模型
   1>卷积层，使用32个3×3的卷积核，激活函数为ReLU。
   2>池化层，使用2×2的池化窗口。
   3>卷积层，使用64个3×3的卷积核，激活函数为ReLU。
   4>池化层，使用2×2的池化窗口。
   5>展平层，将输入的多维数据转换成以为向量。
   6>全连接层，输出维度为64，激活函数是ReLU。
   ```python
encoder = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(embedding_dim, activation="relu"),
    ],
    name="encoder",
)
```
### 5.构建MLP模型
   搭建两个全连接层，输入参数为256，输出类别数，从而实现分类。
   ```python
mlp_model = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(embedding_dim,)),
    layers.Dense(num_classes, activation="softmax")
], name="mlp_model")
```
### 6.准确率
   设置epochs=20，batches=128跑下来准确能够达到0.9179

