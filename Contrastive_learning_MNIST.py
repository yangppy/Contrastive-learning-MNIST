import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
import keras.backend as K
from keras import layers
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')

# 定义超参数
num_classes = 10  # 类别数
input_shape = (28, 28, 1)  # 输入形状
batch_size = 128  # 批量大小
epoches = 10  # 轮次
embedding_dim = 64  # 嵌入维度
alpha = 0.2
path = r'D:\新建文件夹\简历项目\使用对比学习对MNIST数据集进行预训练和分类\mnist.npz'

# 加载并预处理数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data(path)

# 将像素归一化0-1之间
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# 定义损失函数
# 写成闭包是为了传递alpha的值
def contrastive_loss(alpha):
    """
    :param alpha:表示anchor和negative之间为多少时认为他们不匹配
    """
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        # 计算欧式距离
        # K.sqrt(K.sum(K.square(a - b), axis=-1, keepdims=True))
        pos_distance = K.sqrt(K.sum(K.square(anchor - positive), axis=-1))
        neg_distance = K.sqrt(K.sum(K.square(anchor - negative), axis=-1))
        return K.mean((1 - y_true) * K.square(pos_distance) +
                      y_true * K.square(K.maximum(0.0, alpha - neg_distance)))

    return loss


# 自定义DataGenerator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, num_classes, alpha):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.alpha = alpha

    def __len__(self):
        return int(np.ceil(len(self.x)) / float(self.batch_size))

    def __getitem__(self, index):
        batch_x = self.x[index * self.batch_size: (index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size: (index + 1) * self.batch_size]
        anchor = batch_x
        positive = np.zeros_like(anchor)
        negative = np.zeros_like(anchor)

        for i in range(self.batch_size):
            pos_idx = np.random.choice(np.where(self.y == batch_y[i])[0])
            neg_idx = np.random.choice(np.where(self.y != batch_y[i])[0])
            positive[i] = self.x[pos_idx]
            negative[i] = self.x[neg_idx]

        return [anchor, positive, negative], np.zeros((self.batch_size,))


# 构建模型并训练
anchor_input = layers.Input(shape=input_shape, name="anchor_input")
positive_input = layers.Input(shape=input_shape, name="positive_input")
negative_input = layers.Input(shape=input_shape, name="negative_input")

# 建立编码器
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

# 将锚点，正样本，负样本传入编码器
encoded_anchor = encoder(anchor_input)
encoded_positive = encoder(positive_input)
encoded_negative = encoder(negative_input)

merged_output = layers.concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name="merged_layer")
model = keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_output, name="triplet_model")

generator = DataGenerator(x_train.reshape(-1, 28, 28, 1), y_train, batch_size, num_classes, alpha)

# 编译模型
model.compile(loss=contrastive_loss(alpha), optimizer=Adam())
model.fit(generator, epochs=epoches)

# 获取编码器
encoder = model.get_layer("encoder")
# 对测试数据进行编码生成向量
embeddings = encoder.predict(x_test.reshape(-1, 28, 28, 1))
# 使用t-SNE算法对向量进行降维
tsne_embeddings = TSNE(n_components=2).fit_transform(embeddings)

# TSNE可视化处理
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=y_test)
plt.show()

x_train_encoded = encoder.predict(x_train.reshape(-1, 28, 28, 1))
x_test_encoded = encoder.predict(x_test.reshape(-1, 28, 28, 1))

# 训练MLP层并进行分类
mlp_model = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(embedding_dim,)),
    layers.Dense(num_classes, activation="softmax")
], name="mlp_model")

# 编译并训练MLP
mlp_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
mlp_model.fit(x_train_encoded, y_train, batch_size=batch_size, epochs=epoches, validation_data=(x_test_encoded, y_test))

test_loss, test_acc = mlp_model.evaluate(x_test_encoded, y_test)
print("Test accuracy:", test_acc)
