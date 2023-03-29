import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
import keras.backend as K
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 定义超参数
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epoches = 10
path = 'MNIST_data/'

# 加载并预处理数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data(path)

# 将像素归一化0-1之间
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 将标签进行one-hot编码
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 调整图像维度
x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)


# 定义损失函数tf.keras.losses.Loss
def contrastive_loss(y_true, y_pred, margin=1):
    # 计算样本间的欧几里得距离
    distance = K.sqrt(K.sum(K.square(y_pred[0] - y_pred[1]), axis=-1))
    # 计算同类别样本对损失
    loss_same = K.square(distance)
    # 计算异类别样本对损失
    loss_diff = K.square(K.maximum(margin - distance, 0))
    # 计算总损失
    loss = K.mean((1 - y_true) * loss_same + y_true * loss_diff)
    return loss


# 自定义DataGenerator
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, num_classes):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x)) / self.batch_size)

    def __getitem__(self, index):
        # 从数据集中随机选择索引
        indices = np.random.choice(self.indices, size=self.batch_size)

        # 生成正样本和负样本
        anchor_image = self.x[indices]
        positive_indices = np.random.choice(self.num_classes, size=self.batch_size)
        positive_images = np.array([self.x[self.y.argmax(axis=1) == i][j] for i, j in enumerate(positive_indices)])
        negative_indices = np.random.choice(self.num_classes, size=self.batch_size)
        negative_images = np.array([self.x[self.y.argmax(axis=1) == i][j] for i, j in enumerate(negative_indices)])

        # 将图像标签合并到一起
        x = [anchor_image, positive_images, negative_images]
        y = np.zeros(self.batch_size)

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# 定义对比学习模型
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    model = Model(input, x)
    return model


# 生成模型
base_network = create_base_network(input_shape)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
processed_a = base_network(input_a)
processed_b = base_network(input_b)
distance = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True)))([processed_a, processed_b])
model = Model(inputs=[input_a, input_b], outputs=distance)
# 编译并训练模型
train_generator = DataGenerator(x_train, y_train, batch_size, num_classes)
model.compile(loss=contrastive_loss, optimizer='adam')
model.fit_generator(train_generator, epochs=epoches, verbose=1)

# 预训练结束使用TSNE将图像特征可视化
# Extract features from the base network for the test set
test_features = base_network.predict(x_test)

# Apply t-SNE to the features to project them into 2D space
tsne = TSNE(n_components=2, random_state=42)
test_tsne = tsne.fit_transform(test_features)

# Create a scatter plot of the projected features
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.scatter(test_tsne[y_test == i, 0], test_tsne[y_test == i, 1], label=str(i))
plt.legend()
plt.show()
