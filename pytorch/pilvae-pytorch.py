import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import orth
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from torchvision import datasets, transforms

def ActivationFunc(tempH, ActivationFunction, p):
    if ActivationFunction == 'relu':
        H = np.maximum(0, tempH)
    elif ActivationFunction == 'prelu':
        alpha = 0.02
        H = np.maximum(alpha * tempH, tempH)
    elif ActivationFunction == 'gelu':
        H = tempH * 1.0 / (1 + np.exp(-p * tempH * 1.702))
    elif ActivationFunction == 'sigmoid':
        H = 1.0 / (1 + np.exp(-tempH))
    elif ActivationFunction == 'sin':
        H = np.sin(tempH)
    elif ActivationFunction == 'tanh':
        H = np.tanh(tempH)
    return H

def ppcamle(data, q):
    N, d = data.shape
    mu = np.mean(data, axis=0)
    T = data - mu
    S = T.T.dot(T) / N
    D, V = np.linalg.eigh(S)
    sorted_indices = np.argsort(D)[::-1]
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    sigma = np.sum(D[(q + 1):]) / (d - q)
    Uq = V[:, :q]
    lambda_q = D[:q]
    w = Uq.dot(np.sqrt(np.diag(lambda_q) - sigma * np.identity(q)))
    C = w.dot(w.T) + sigma * np.identity(d)
    
    det_C = np.linalg.det(C)
    if det_C < 1e-10:
        det_C = 1e-10
    L = -N * (d * np.log(2 * np.pi) + np.log(det_C) + np.trace(np.linalg.solve(C, S)) / 2)
    return mu, w, sigma, L

latent_dim = 12
MAX_EN_LAYER = 2
mean_value = 0
sig = 1e-2
para = 1
actFun = 'prelu'
lambda0 = 1e-3
vae = []
HiddenO = []
l = 1
value_to_find = 6# 查找的索引值
num_training_samples = 2000 # 训练样本量


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_images = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()
test_images = test_dataset.data.numpy()
test_labels = test_dataset.targets.numpy()
print("合并前的数据集形状:", train_images.shape, train_images.shape)

all_images = np.concatenate((train_images, test_images), axis=0)
all_labels = np.concatenate((train_labels, test_labels), axis=0)
print("合并后的数据集形状:", all_images.shape, all_labels.shape)
train_labels
type(train_labels)
# 使用numpy.min()函数找到最小值
min_value = np.min(train_labels)
# 使用numpy.max()函数找到最大值
max_value = np.max(train_labels)
print("最小值:", min_value, "最大值:", max_value)
indices = np.where(train_labels == value_to_find)
# 打印索引元组
print(f"值为{value_to_find}的索引:", indices)
type(train_images)
# 使用切片操作选中指定序号的图像
selected_images = train_images[indices]

# 打印结果
print("选中的图像形状:", selected_images.shape)
len(indices[0])
# 使用NumPy生成随机整数序列
ridx = np.random.randint(0, selected_images.shape[1] - 1, num_training_samples)
X_train = selected_images[ridx]

# 将后两维合并成一个二维数组，保持第一维不变
X_train_2d = X_train.reshape(X_train.shape[0], -1)  # 每行表示一个观察或样本，每列表示一个特征或变量
# 打印结果
print("原始三维数组形状:", X_train.shape)
print("合并后的二维数组形状:", X_train_2d.shape)
rank_data = np.linalg.matrix_rank(X_train_2d)
print("输入的秩：", rank_data)
input_dim = X_train_2d.shape[1]  # dim of input data
hidden_dim = X_train_2d.shape[0]  # Gn-PIL dim of H

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))  # 映射到 0-1 之间
X_train_2d = scaler.fit_transform(X_train_2d)
InputLayer = X_train_2d.T
print("InputLayer's shape:", InputLayer.shape)

while l <= MAX_EN_LAYER:
    # %%%%%%%%%% Encoder %%%%%%%%%%
    # %%%%%% 1st layer: PIL0
    # 使用NumPy创建具有随机值的二维数组
    InputWeight = np.random.randn(hidden_dim, input_dim)
    if hidden_dim >= input_dim:
        InputWeight = orth(InputWeight)
    else:
        InputWeight = orth(InputWeight.T).T
    # Compute the rank of the matrix InputLayer
    print(InputWeight.shape)
    print(InputLayer.shape)
    matrix_rank = np.linalg.matrix_rank(InputLayer)
    print("rank of InputLayer：", matrix_rank)

    print("inputweight:", InputWeight)
    tempH = InputWeight.dot(InputLayer)
    H1 = ActivationFunc(tempH, actFun, para)
    vae.append(InputWeight)  # vae{l}.WI = InputWeight
    HiddenO.append(H1)
    l = l + 1

    InputLayer = H1
    hidden_dim = InputLayer.shape[1]
    input_dim = InputLayer.shape[0]
    print("InputLayer shape:", InputLayer.shape)
InputLayer_pinv = np.linalg.pinv(InputLayer)
# Compute the rank of the matrix InputLayer
matrix_rank = np.linalg.matrix_rank(InputLayer_pinv)
print("rank of InputLayer：", matrix_rank)

# 生成具有指定均值和标准差的随机数，大小与 InputLayer_pinv 相同
random_noise = np.random.normal(mean_value, sig, size=InputLayer_pinv.shape)

# 将随机噪声添加到 InputLayer_pinv 中
InputLayer_pinv = InputLayer_pinv + random_noise

# Compute the rank of the matrix InputLayer
matrix_rank = np.linalg.matrix_rank(InputLayer_pinv)
print("rank of matrix after GnPIL：", matrix_rank)

vae.append(InputLayer_pinv)
tempH = InputLayer_pinv.dot(InputLayer)
H2 = ActivationFunc(tempH, actFun, para)
HiddenO.append(H2)
l = l + 1
Y = H2.T
H2.shape




mu, w, sigma, L = ppcamle(Y, latent_dim)
print("sigma:", sigma)
# 计算 Z  Z 是一个 (latent_dim, num_trainingSamples) 的矩阵
Z = np.linalg.solve(w.T.dot(w) + sigma * np.identity(latent_dim), w.T).dot(
    Y - np.tile(mu.T, (num_training_samples, 1)).T)
print("shape of Z:", Z.shape)
vae.append(w)
l = l + 1
# Decoder
# 1st layer: Using Z to reconstruct H2
ZZT = Z.dot(Z.T)
# OutputWeight = H2.dot(Z.T) / (ZZT + lambda0 * np.eye(latent_dim))
OutputWeight = H2.dot(Z.T).dot(np.linalg.pinv(ZZT + lambda0 * np.eye(latent_dim)))
print("layer of rec H2:", l)
vae.append(OutputWeight)
tempH = OutputWeight.dot(Z)
rec_H2 = tempH
l = l + 1
print(l)
# 2nd layer: Using H2 to reconstruct H1
dl = l - 4
while dl > 0:
    # 计算 OutputWeight
    tempH_transpose = tempH.T
    tempHHT = tempH.dot(tempH_transpose)
    eye_matrix = lambda0 * np.eye(hidden_dim)
    #     OutputWeight = HiddenO[dl - 1].dot(tempH_transpose) / (tempHHT + eye_matrix)
    OutputWeight = (HiddenO[dl - 1].dot(tempH_transpose)).dot(np.linalg.pinv(tempHHT + eye_matrix))
    vae.append(OutputWeight)  # 5

    # 更新 tempH
    tempH = OutputWeight.dot(tempH)

    # 更新 rec_H1 (如果需要)
    rec_H1 = tempH  # 此行可以根据需要添加

    l += 1
    dl -= 1
# 3rd layer:  Using H1 to reconstruct X
rec_H1H1T = rec_H1.dot(rec_H1.T)
lambda_eye = lambda0 * np.eye(hidden_dim)

# 计算 OutputWeight
# OutputWeight = X_train_2d.dot(rec_H1.T) / (rec_H1H1T + lambda_eye)
OutputWeight = (X_train_2d.T.dot(rec_H1.T)).dot(np.linalg.pinv(rec_H1H1T + lambda_eye))
vae.append(OutputWeight)  # 6

# 计算 tempX
tempX = OutputWeight.dot(rec_H1)

# 更新 rec_X
rec_X = tempX
rec_X.T.shape
rec_X.T[0]
type(rec_X.T[0])


# 使用imshow函数将数组绘制成图像
plt.imshow(np.abs(rec_X.T[0].reshape((28, 28))), cmap='gray')  # cmap参数指定了使用灰度颜色映射

# 显示图像
plt.show()
num_samples_test = 1000
# value_to_find = 2
# 使用numpy.where()查找值为2的索引
indices = np.where(test_labels == value_to_find)

# 打印索引元组
print("值为2的索引:", indices)
# 使用切片操作选中指定序号的图像
selected_TestImages = test_images[indices]

# 打印结果
print("选中的图像形状:", selected_TestImages.shape)
len(indices[0])
# 使用NumPy生成随机整数序列
ridx = np.random.randint(0, selected_TestImages.shape[1] - 1, num_samples_test)
X_test = selected_TestImages[ridx]

# 将后两维合并成一个二维数组，保持第一维不变
X_test_2d = X_test.reshape(X_test.shape[0], -1)  # 每行表示一个观察或样本，每列表示一个特征或变量

# 打印结果
print("原始三维数组形状:", X_test.shape)
print("合并后的二维数组形状:", X_test_2d.shape)
X_test_2d = scaler.fit_transform(X_test_2d)

InputLayer = X_test_2d.T
print("InputLayer's shape:", InputLayer.shape)
l = 1
while l <= MAX_EN_LAYER:
    tempH_test1 = np.dot(vae[l - 1], InputLayer)
    H_test1 = ActivationFunc(tempH_test1, actFun, para)
    InputLayer = H_test1
    l += 1
tempH_test2 = np.dot(vae[l - 1], InputLayer)
H_test2 = ActivationFunc(tempH_test2, actFun, para)
InputLayer = H_test2
l += 1
print(l)
print("mu's shape:", mu.T.shape)
InputLayer_mu = InputLayer - np.tile(mu, (num_samples_test, 1)).T
Z_test = (vae[l - 1].T.dot(vae[l - 1]) + sigma * np.eye(latent_dim)).dot(np.linalg.pinv(vae[l - 1])).dot(InputLayer_mu)
l = l + 1
print(l)
H_rec_test2 = np.dot(w, Z_test) + np.tile(mu.T, (num_samples_test, 1)).T
l = l + 1
print(l)
dl = l - 4
while dl > 0:
    H_rec_test1 = vae[l - 1].dot(H_rec_test2)

    l += 1
    dl -= 1
X_rec_test = vae[l - 1].dot(H_rec_test1)
print("RecTest's shape", X_rec_test.shape)
# 设置图形大小
plt.figure(figsize=(8, 8))
#     遍历每张图片并将其添加到图形子集中
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(np.abs(X_rec_test.T[i].reshape((28, 28))), cmap='gray')  # 假设这是灰度图像
    plt.axis('off')  # 关闭坐标轴

# 显示图形
plt.show()


# Generating new sample
num_samples = 1000
latent_samples = np.random.randn(latent_dim, num_samples)
# Generate the covariance matrix
print(sigma)
covariance = sigma * np.eye(hidden_dim)
# Generate noise from the Gaussian distribution
# ns = mvnrnd(zeros(num_samples, hidden_dim), covariance, num_samples);
ns = multivariate_normal.rvs(mean=np.zeros(hidden_dim), cov=covariance, size=num_samples)
print("ns's shape:", ns.shape)
# Generate new samples
hidden_samples1 = w.dot(latent_samples) + np.tile(mu.T, (num_samples, 1)).T
l = 4 + MAX_EN_LAYER
generated_samples2 = vae[l - 1].dot(hidden_samples1)
temprecH = generated_samples2
while l < 4 + MAX_EN_LAYER * 2:
    l = l + 1
    temprecH = vae[l - 1].dot(temprecH)

generated_X = temprecH
print("generated_X:", generated_X.shape)
# 使用imshow函数将数组绘制成图像
plt.imshow(np.abs(generated_X.T[0].reshape((28, 28))), cmap='gray')  # cmap参数指定了使用灰度颜色映射

# 显示图像
plt.show()
# 设置图形大小
plt.figure(figsize=(8, 8))
#     遍历每张图片并将其添加到图形子集中
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(np.abs(generated_X.T[i].reshape((28, 28))), cmap='gray')  # 假设这是灰度图像
    plt.axis('off')  # 关闭坐标轴

# 显示图形
plt.show()
