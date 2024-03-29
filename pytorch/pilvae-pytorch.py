import numpy as np
import torch
import matplotlib
from torch.distributions import multivariate_normal

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 参数定义
para = 1
actFun = 'prelu'  # prelu gau
batch_size = 16
l = 1
epochs = 3
mean_value = 0
sig = 1e-2
latent_dim = 15
lambda0 = 1e-3
vae = []
HiddenO = []
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False)

# 遍历 DataLoader 中的每一个 batch
target_label = 2  # 筛选的标签值
num_samples = 6000  # 需要抽取的样本数量
target_data = []
for batch_data, batch_label in train_loader:
    # 筛选和过滤指定标签的数据
    mask = torch.eq(batch_label, target_label)
    indices = torch.nonzero(mask).squeeze()
    target_batch_data = torch.index_select(batch_data, 0, indices)

    # 将每一个 batch 的指定标签的数据添加到结果列表中
    target_data.append(target_batch_data)
# 拼接所有的数据
target_data = torch.cat(target_data, dim=0)

# 如果筛选出的数据量多于需要的数量，随机抽样
if target_data.size(0) >= num_samples:
    # 创建随机索引
    indices = torch.randperm(target_data.size(0))[:num_samples]
    # 根据索引抽取数据
    target_data = target_data[indices]

# 确保最后形状符合要求，并处理可能的情况：筛选出的数据不足 num_samples
actual_num_samples = min(num_samples, target_data.size(0))
X_train_2d = target_data.reshape(actual_num_samples, -1)
X_train_2d = (X_train_2d + 1) / 2
input_size = X_train_2d.shape[1]  # dim of input data
hidden_size = X_train_2d.shape[0]


def ActivationFunc(tempH, ActivationFunction, p):
    if ActivationFunction == 'relu':
        #         tempH[tempH <= 0] = 0
        #         tempH[tempH > 0] = tempH
        #         H = tempH
        H = np.maximum(0, tempH)
    elif ActivationFunction == 'prelu':
        alpha = 0.02;
        #         tempH[tempH <= 0] = alpha*tempH;
        #         H = tempH
        H = np.maximum(alpha * tempH, tempH)
    elif ActivationFunction == 'gelu':  # xσ(1.702x)
        H = tempH * 1.0 / (1 + np.exp(-p * tempH * 1.702))
    elif ActivationFunction == 'sigmod':
        H = 1.0 / (1 + np.exp(-tempH))
    elif ActivationFunction == 'srelu':
        tempH[tempH <= 0] = 0
        tempH[tempH > 0] = tempH
        H = tempH
    elif ActivationFunction == 'sin':
        H = np.sin(tempH)
    elif ActivationFunction == 'tanh':
        H = np.np.tanh(tempH)
    return H


def torchOrth(A):
    # Perform Singular Value Decomposition
    u, s, v = torch.linalg.svd(A, full_matrices=False)
    # Normalize each column to turn them into unit vectors
    norm_u = u / u.norm(dim=0)
    return norm_u


def PIL0(InputLayer, input_size, hidden_size, l):
    InputWeight = torch.randn(hidden_size, input_size)
    if hidden_size >= input_size:
        InputWeight = torchOrth(InputWeight)
    else:
        InputWeight = torchOrth(InputWeight.T).T
    matrix_rank = torch.linalg.matrix_rank(InputLayer)
    tempH = torch.mm(InputWeight, InputLayer)
    H1 = ActivationFunc(tempH, actFun, para)
    l = l + 1
    InputLayer = H1
    hidden_size = InputLayer.shape[1]
    input_size = InputLayer.shape[0]
    vae.append(InputWeight)
    HiddenO.append(H1)
    return InputLayer, hidden_size, input_size, l


def Gn_PIL(InputLayer, input_size, hidden_size, l):
    InputLayer_pinv = torch.pinverse(InputLayer)
    noisy = torch.normal(mean_value, sig, size=InputLayer.shape)
    InputLayer_pinv = InputLayer + noisy
    tempH = torch.matmul(InputLayer_pinv, InputLayer)
    H2 = ActivationFunc(tempH, actFun, para)
    l = l + 1
    vae.append(InputLayer_pinv)
    HiddenO.append(H2)
    return H2, l


def ppcamle(data, q):
    # data: 输入数据，N行d列
    # q: 潜在空间的维度
    N, d = data.shape
    mu = torch.mean(data, dim=0)
    T = data - mu  # Broadcasting automatically expands mu
    S = T.t().mm(T) / N  # S = T' * T / N
    D, V = torch.linalg.eig(S)  # Eigenvalue decomposition, 注意：返回的特征值是复数
    D = torch.real(D)  # 取特征值的实部
    V = torch.real(V)  # 确保特征向量同样是实数类型
    sorted_indices = torch.argsort(D, descending=True)  # 降序排序特征值
    D = D[sorted_indices]
    V = V[:, sorted_indices]
    sigma = torch.sum(D[q:]) / (d - q)  # 计算σ
    Uq = V[:, :q]  # 提取前q个特征向量
    lambda_q = D[:q]  # 提取前q个特征值
    diag_lambda_q = torch.diag(lambda_q - sigma)  # 构建对角矩阵 lambda_q - sigma*I
    w = torch.mm(Uq, torch.sqrt(diag_lambda_q))  # w = Uq * sqrt(lambda_q - sigma*I)
    C = w.mm(w.t()) + sigma * torch.eye(d)  # 计算协方差矩阵 C
    log_2pi = torch.log(torch.tensor(2.0 * np.pi))
    sign, log_det_C = torch.linalg.slogdet(C)
    assert sign > 0  # 确保行列式为正
    invC_S = torch.linalg.solve(C, S)
    trace_invC_S = torch.trace(invC_S)
    L = -N * (d * log_2pi + log_det_C + trace_invC_S / 2.0)
    return mu, w, sigma, L


def ppca(H2, l, latent_dim=latent_dim):
    mu, w, sigma, L = ppcamle(H2, latent_dim)
    # 计算逆矩阵的一部分
    A = w.T @ w + sigma * torch.eye(latent_dim)
    # 计算逆矩阵
    A_inv = torch.inverse(A)
    mu_expanded = mu.unsqueeze(1).expand_as(H2.T)
    # 应用公式计算 Z
    Z = A_inv @ w.T @ (H2.T - mu_expanded)
    vae.append(w)
    return Z, l, sigma


def Zrec(Z, H2, l, lambda0=lambda0):
    ZZT = torch.matmul(Z, Z.T)
    # 计算 H2 和 Z.T 的点积
    HZT = torch.matmul(H2, Z.T)
    # 创建一个与 ZZT 相同维度的单位矩阵，并乘以lambda0
    I = lambda0 * torch.eye(latent_dim, dtype=ZZT.dtype, device=ZZT.device)
    # 计算括号内的矩阵和它的伪逆
    pinv = torch.pinverse(ZZT + I)
    # 最后计算 OutputWeight
    OutputWeight = torch.matmul(HZT, pinv)
    tempH = torch.matmul(OutputWeight, Z)
    l = l + 1
    vae.append(OutputWeight)
    return tempH, l


def H2rec(tempH, l, dl, lambda0=lambda0, hidden_size=hidden_size):
    tempH_T = tempH.T
    tempHHT = torch.matmul(tempH, tempH.T)
    eye_matrix = lambda0 * torch.eye(hidden_size)
    step1 = torch.matmul(HiddenO[dl - 1], tempH_T)
    # numpy.linalg.pinv(tempHHT + eye_matrix)
    # 先添加单位矩阵到 tempHHT:
    regularized = tempHHT + eye_matrix  # 这里假设 eye_matrix 已经是像 torch.eye 生成的单位矩阵
    # 计算伪逆
    step2 = torch.pinverse(regularized)
    # 现在进行最终的点乘操作得到 OutputWeight:
    OutputWeight = torch.matmul(step1, step2)
    tempH = torch.matmul(OutputWeight, tempH)
    l = l + 1
    dl = dl - 1
    return tempH, l, dl


def H1rec(tempH, X_train_2d, hidden_size=hidden_size, lambda0=lambda0):
    tempHHT = torch.matmul(tempH, tempH.T)
    lambda_eye = lambda0 * torch.eye(hidden_size)
    X_train_2d_rec_H1_T = torch.matmul(X_train_2d.T, tempH.T)
    # 计算 rec_H1H1T 和 lambda_eye 的和
    rec_H1H1T_lambda_eye = tempHHT + lambda_eye
    # 接着计算伪逆
    rec_H1H1T_lambda_eye_pinv = torch.pinverse(rec_H1H1T_lambda_eye)
    # 最后计算 OutputWeight
    OutputWeight = torch.matmul(X_train_2d_rec_H1_T, rec_H1H1T_lambda_eye_pinv)
    vae.append(OutputWeight)
    tempH = torch.matmul(OutputWeight, tempH)
    return tempH


X_train_3d = X_train_2d
X_train_2d = X_train_2d.T
print(X_train_3d.shape)
print(X_train_2d.shape)
for epoch in range(epochs):
    X_train_2d, input_size, hidden_size, l = PIL0(X_train_2d, input_size, hidden_size, l)
    print(X_train_2d.shape)
    print(0)

H2, l = Gn_PIL(X_train_2d, input_size, hidden_size, l)
print(H2.shape)
H2 = H2.T
Z, l, sigma = ppca(H2, l)
print(Z.shape)
rec_H2, l = Zrec(Z, H2, l)
print(rec_H2.shape)
print(l)
dl = l - 3
while dl>0:
    rec_H1, l, dl = H2rec(rec_H2, l, dl)
    print(rec_H1.shape)
    print(1)
rec_X = H1rec(rec_H1, X_train_3d)
print(rec_X.T.shape)
# print(rec_X.T)
rec_X_np = rec_X.T[2].detach().numpy()
# 用reshape方法改变数组形状，而不是view
rec_X_np_reshaped = rec_X_np.reshape(28, 28)
# 显示图像，abs确保所有数值是非负的，cmap指定色彩映射
plt.imshow(np.abs(rec_X_np_reshaped), cmap='gray')
plt.show()  # 显示图像


# # Generating new sample
# num_samples = 1000
# latent_samples = np.random.randn(latent_dim, num_samples)
# # Generate the covariance matrix
# print(sigma)
# covariance = sigma * np.eye(hidden_size)
# # Generate noise from the Gaussian distribution
# # ns = mvnrnd(zeros(num_samples, hidden_dim), covariance, num_samples);
# ns = multivariate_normal.rvs(mean=np.zeros(hidden_size), cov=covariance, size=num_samples)
# print("ns's shape:", ns.shape)
# # Generate new samples
# hidden_samples1 = w.dot(latent_samples) + np.tile(mu.T, (num_samples, 1)).T
# l = 4 + MAX_EN_LAYER
# generated_samples2 = vae[l - 1].dot(hidden_samples1)
# temprecH = generated_samples2
# while l < 4 + MAX_EN_LAYER * 2:
#     l = l + 1
#     temprecH = vae[l - 1].dot(temprecH)
#
# generated_X = temprecH
# print("generated_X:", generated_X.shape)
# # 使用imshow函数将数组绘制成图像
# plt.imshow(np.abs(generated_X.T[0].reshape((28, 28))), cmap='gray')  # cmap参数指定了使用灰度颜色映射
#
# # 显示图像
# plt.show()
# # 设置图形大小
# plt.figure(figsize=(8, 8))
# #     遍历每张图片并将其添加到图形子集中
# for i in range(100):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(np.abs(generated_X.T[i].reshape((28, 28))), cmap='gray')  # 假设这是灰度图像
#     plt.axis('off')  # 关闭坐标轴
#
# # 显示图形
# plt.show()


