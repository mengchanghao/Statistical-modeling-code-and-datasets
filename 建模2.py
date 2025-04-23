#############################
# 1. 数据预处理模块（修正：统一特征维度）
#############################
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import matplotlib.pyplot as plt

# 设置中文字体（以 SimHei 为例）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

class DataPreprocessor:
    def __init__(self, wdbc_path, wpbc_path):
        self.wdbc_path = wdbc_path
        self.wpbc_path = wpbc_path

        # 统一特征列名（修正：确保两个数据集处理后的特征数量一致）
        self.common_features = [
            'Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean', 'Smoothness_mean',
            'Compactness_mean', 'Concavity_mean', 'Concave_points_mean', 'Symmetry_mean',
            'Fractal_dimension_mean', 'Radius_se', 'Texture_se', 'Perimeter_se', 'Area_se',
            'Smoothness_se', 'Compactness_se', 'Concavity_se', 'Concave_points_se',
            'Symmetry_se', 'Fractal_dimension_se', 'Radius_worst', 'Texture_worst',
            'Perimeter_worst', 'Area_worst', 'Smoothness_worst', 'Compactness_worst',
            'Concavity_worst', 'Concave_points_worst', 'Symmetry_worst', 'Fractal_dimension_worst'
        ]  # 共30个特征

        self.wpbc_extra_cols = ['ID', 'Diagnosis', 'Time', 'Status']

    def load_data(self):
        # 读取时仅保留共同特征（修正：统一维度）
        self.wdbc = pd.read_csv(self.wdbc_path, header=None)
        self.wdbc = self.wdbc.iloc[:, 2:]  # 前两列是ID和Diagnosis

        self.wpbc = pd.read_csv(self.wpbc_path, header=None)
        # WPBC数据需要对齐到30维：原始数据列数较多，按位置选择对应特征
        self.wpbc = self.wpbc.iloc[:, 2:32]  # 假设特征列位置与WDBC一致

    def preprocess(self):
        # 源域处理
        imputer = SimpleImputer(strategy='mean')
        self.qt = QuantileTransformer(
            n_quantiles=100,
            output_distribution='normal',
            random_state=42
        )

        # WDBC处理
        self.wdbc_imputed = pd.DataFrame(imputer.fit_transform(self.wdbc), columns=self.common_features)
        self.wdbc_scaled = self.qt.fit_transform(self.wdbc_imputed)
        # WPBC处理
        self.wpbc_imputed = pd.DataFrame(imputer.fit_transform(self.wpbc), columns=self.common_features)
        self.wpbc_scaled = self.qt.transform(self.wpbc_imputed)

        # 目标变量处理（假设最后一列是Time）
        self.wpbc_target = self.wpbc.iloc[:, -1].copy()
        self.wpbc_target_imputed = self.wpbc_target.fillna(self.wpbc_target.mean())
        return self.wdbc_scaled, self.wpbc_scaled, self.wpbc_target_imputed

    def show_preprocessed_data(self):
        print("wdbc_features_scaled (源域) shape:", self.wdbc_scaled.shape)
        print("wpbc_features_scaled (目标域) shape:", self.wpbc_scaled.shape)
        print("wpbc_target_imputed shape:", self.wpbc_target_imputed.shape)

#############################
# 2. 特征对齐与知识迁移模块（修正：设备兼容性）
#############################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super(VAE, self).__init__()

        # 定义编码器和解码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # 增加层宽度
            nn.LeakyReLU(0.2),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim)  # 新增批归一化
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 使用 Xavier 初始化
        self.apply(self.weights_init)

    def weights_init(self, m):
        """Xavier初始化函数"""
        if isinstance(m, nn.Linear):  # 只针对线性层进行初始化
            nn.init.xavier_uniform_(m.weight)  # 使用 Xavier 均匀分布初始化权重
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # 将偏置初始化为0

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z





















class DomainDiscriminator(nn.Module):
    """改进后的领域判别器 (AMP兼容)"""
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim//2, 1)
            # 注意：移除了最后的 Sigmoid 层
        )

    def forward(self, z):
        """直接输出 logits (未经 Sigmoid)"""
        return self.net(z)









def conditional_diffusion_alignment(z_source, z_target, pseudo_labels,
                                    iterations=600, step_size=0.1, temp=0.5, alpha=0.95):
    """
    基于伪标签的条件扩散对齐
    :param z_source: 源域隐变量 (B, D)
    :param z_target: 目标域隐变量 (B, D)
    :param pseudo_labels: 伪标签 (B,)
    :param temp: 初始温度系数
    :param alpha: 温度衰减系数
    """
    aligned_z = z_source.clone()
    unique_labels = torch.unique(pseudo_labels)

    # 初始化置信度权重
    label_confidences = torch.ones_like(pseudo_labels, dtype=torch.float32)

    for label in unique_labels:
        mask = (pseudo_labels == label)
        if mask.sum() == 0:
            continue

        target_center = z_target[mask].mean(dim=0)

        for _ in range(iterations):
            sim = F.cosine_similarity(aligned_z[mask], target_center.unsqueeze(0), dim=1)
            adj_step = step_size * torch.sigmoid(sim / temp).unsqueeze(1)

            # 扩散步骤：向目标中心迁移
            aligned_z[mask] += adj_step * (target_center - aligned_z[mask])

            # 动态调整温度
            temp = max(0.1, temp * alpha)

            # 更新置信度权重（示例：根据相似度调整）
            label_confidences[mask] = torch.sigmoid(sim)

            # 应用置信度加权
            aligned_z[mask] += label_confidences[mask].unsqueeze(1) * adj_step * (target_center - aligned_z[mask])

    return aligned_z



def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """最大均值差异（MMD）损失计算"""
    total = torch.cat([source, target], dim=0)
    n_samples = total.size(0)

    # 计算成对距离矩阵
    total = total.view(n_samples, -1)
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
    diff = total0 - total1
    diff = torch.sum(diff  **  2, dim = 2)  # 修正维度为2

    # 动态带宽计算
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(diff) / (n_samples  **  2 - n_samples + 1e-8)
    bandwidth /= kernel_mul **  (kernel_num // 2)

    # 多核MMD计算
    kernel_val = torch.zeros_like(diff)
    for i in range(kernel_num):
        bandwidth_k = bandwidth * (kernel_mul  ** i)
        kernel_val += torch.exp(-diff / (bandwidth_k + 1e-8))

    # 分解核矩阵
    k_ss = kernel_val[:source.size(0), :source.size(0)]
    k_tt = kernel_val[source.size(0):, source.size(0):]
    k_st = kernel_val[:source.size(0), source.size(0):]

    # 最终MMD值
    mmd = (k_ss.mean() + k_tt.mean() - 2 * k_st.mean())
    return mmd



def compute_total_loss(recon_loss, kl_loss, mmd_loss, adv_loss, epoch, num_epochs):
    """
    动态调整各个损失的权重，确保损失协同训练。
    :param recon_loss: 重建损失
    :param kl_loss: KL 散度损失
    :param mmd_loss: MMD 损失
    :param adv_loss: 对抗损失
    :param epoch: 当前训练的 epoch 数
    :param num_epochs: 总训练周期数
    :return: 总损失
    """
    # 动态调整各个损失的权重
    recon_weight = 0.7
    mmd_weight = 0.2
    adv_weight = 0.5 * (1 - epoch / num_epochs)  # 增强对抗训练权重，后期提升对抗损失的影响

    # 计算总损失
    total_loss = recon_weight * recon_loss + kl_loss + mmd_weight * mmd_loss + adv_weight * adv_loss
    return total_loss




def train_feature_alignment(source_data, target_data, device, num_epochs=20, batch_size=64,
                            latent_dim=15, lr=1e-3, mmd_weight=1.5, adv_weight=0.5):
    """整合改进的特征对齐训练"""
    # 设备检查
    source_data = source_data.to(device)
    target_data = target_data.to(device)
    input_dim = source_data.size(1)

    # 模型初始化
    vae = VAE(input_dim, latent_dim).to(device)
    discriminator = DomainDiscriminator(latent_dim).to(device)

    # 优化器配置
    optimizer = optim.AdamW([
        {'params': vae.parameters(), 'lr': lr},
        {'params': discriminator.parameters(), 'lr': lr * 0.1}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # 数据加载器（确保有效批次）
    source_loader = DataLoader(TensorDataset(source_data), batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(TensorDataset(target_data), batch_size=batch_size, shuffle=True, drop_last=True)

    # 混合精度训练初始化
    scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    recon_hist, mmd_hist, adv_hist = [], [], []
    for epoch in range(num_epochs):
        total_loss = 0.0
        vae.train()
        discriminator.train()

        # 使用zip_longest处理批次长度不一致
        for batch_src, batch_tgt in zip(source_loader, target_loader):
            try:
                src_batch = batch_src[0].to(device)
                tgt_batch = batch_tgt[0].to(device)

                # ================== 混合精度前向传播 ==================
                with torch.cuda.amp.autocast():
                    # VAE编码
                    recon_src, mu_src, logvar_src, z_src = vae(src_batch)
                    recon_tgt, mu_tgt, logvar_tgt, z_tgt = vae(tgt_batch)

                    # ========== 条件扩散对齐（每5个epoch执行） ==========
                    if epoch % 5 == 0:
                        with torch.no_grad():
                            # 生成伪标签
                            pseudo_labels = cluster_predict(z_src, z_tgt)

                            # 执行条件扩散对齐
                            aligned_z = conditional_diffusion_alignment(
                                z_src, z_tgt,
                                pseudo_labels=pseudo_labels,
                                iterations=300,
                                step_size=0.1
                            )
                            z_src = aligned_z

                    # ========== 损失计算 ==========

                    # 重建损失
                    recon_loss = 0.7 * F.mse_loss(recon_src, src_batch) + 0.3 * F.mse_loss(recon_tgt, tgt_batch)

                    # KL散度
                    kl_div = -0.5 * torch.sum(
                        1 + logvar_src - mu_src.pow(2) - logvar_src.exp()
                    ) / (batch_size * input_dim)

                    # 直接传入logits，无需手动Sigmoid
                    logits_src = discriminator(z_src.detach())
                    logits_tgt = discriminator(z_tgt.detach())

                    # 动态调整对抗损失权重
                    current_adv_weight = adv_weight * (1 - epoch / num_epochs)
                    adv_loss = current_adv_weight * (
                        F.binary_cross_entropy_with_logits(logits_src, torch.ones_like(logits_src)) +
                        F.binary_cross_entropy_with_logits(logits_tgt, torch.zeros_like(logits_tgt))
                    )

                    # 计算MMD损失
                    mmd = mmd_weight * mmd_loss(z_src, z_tgt)

                    # 计算总损失，损失函数协同训练
                    total_loss = compute_total_loss(recon_loss, kl_div, mmd, adv_loss, epoch, num_epochs)

                    # 记录损失
                    recon_hist.append(recon_loss.item())
                    mmd_hist.append(mmd.item())
                    adv_hist.append(adv_loss.item())

                # ================== 反向传播 ==================
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss += total_loss.item()

            except Exception as e:
                print(f"批次处理异常: {str(e)}，跳过当前批次")
                continue

        # 更新学习率
        scheduler.step(total_loss)
        # 打印训练信息
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {total_loss / len(source_loader):.4f}")

    # 绘制训练过程中的损失曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3))
    plt.plot(recon_hist, label="Recon")
    plt.plot(mmd_hist, label="MMD")
    plt.plot(adv_hist, label="ADV")
    plt.legend()
    plt.title("对齐阶段损失曲线")
    plt.show()

    # t-SNE可视化
    try:
        from sklearn.manifold import TSNE
        z_samp = torch.cat([z_src.detach(), z_tgt.detach()]).cpu().numpy()
        tsne = TSNE(perplexity=30, random_state=42).fit_transform(z_samp)
        n_src = z_src.size(0)
        plt.figure(figsize=(6, 5))
        plt.scatter(tsne[:n_src, 0], tsne[:n_src, 1], alpha=.5, label='source')
        plt.scatter(tsne[n_src:, 0], tsne[n_src:, 1], alpha=.5, label='target')
        plt.legend()
        plt.title("隐空间对齐 t‑SNE")
        plt.show()
    except Exception as _:
        pass

    return vae, discriminator



















from sklearn.cluster import MiniBatchKMeans  # 添加这行
def cluster_predict(z_src, z_tgt):
    """批次敏感的伪标签生成"""
    # 确保输入为当前批次
    assert z_src.size(0) == z_tgt.size(0), "批次大小必须一致"

    combined = torch.cat([z_src, z_tgt]).cpu().numpy()
    km = MiniBatchKMeans(n_clusters=2, batch_size=combined.shape[0] // 2)
    labels = torch.from_numpy(km.fit_predict(combined)).to(z_src.device)

    # 分离源域标签
    return labels[:z_src.size(0)]



#############################
# 3. 伪标签生成与先验构建（修正：异常处理）
#############################
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso

from sklearn.semi_supervised import LabelSpreading


def generate_pseudolabels(z_aligned, threshold=0.9, min_sil_score=0.15):
    """增强版伪标签生成，包含自动参数调整"""
    from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    import warnings

    # 禁用不必要的警告
    warnings.filterwarnings("ignore", category=UserWarning)

    z_np = z_aligned.cpu().numpy() if isinstance(z_aligned, torch.Tensor) else z_aligned
    best_labels = None
    best_score = -1

    # ==================== 改进点 1：优化聚类参数 ====================
    clustering_configs = [
        # 高斯混合模型
        (GaussianMixture, {'n_components': 2, 'covariance_type': 'diag'}),
        (GaussianMixture, {'n_components': 3, 'reg_covar': 1e-3}),
        # K-Means
        (KMeans, {'n_clusters': 2, 'n_init': 20}),
        (KMeans, {'n_clusters': 3, 'n_init': 20}),
        # 谱聚类
        (SpectralClustering, {'n_clusters': 2, 'affinity': 'nearest_neighbors', 'n_neighbors': 10}),
        (SpectralClustering, {'n_clusters': 3, 'gamma': 0.1}),
        # DBSCAN（新增）
        (DBSCAN, {'eps': 0.5, 'min_samples': 5})
    ]

    # ==================== 改进点 2：并行尝试多种配置 ====================
    for config in clustering_configs:
        model_class, params = config
        try:
            model = model_class( ** params)
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(z_np)
            else:
                model.fit(z_np)
                labels = model.labels_

            # 过滤无效标签（新增异常值处理）
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2 or (labels == -1).any():  # DBSCAN可能有-1标签
                continue

            # 计算轮廓系数时排除噪声点（针对DBSCAN）
            valid_mask = labels != -1 if hasattr(model, 'eps') else slice(None)
            score = silhouette_score(z_np[valid_mask], labels[valid_mask])

            if score > best_score:
                best_score = score
                best_labels = labels
        except:
            continue

    # ==================== 改进点 3：优化备用方案 ====================
    if best_score < min_sil_score or best_labels is None:
        print(f"⚠️ 所有聚类方法失败(sil_score={best_score:.2f})，启用密度连通性方案")

        # 改进的密度连通性方案
        nbrs = NearestNeighbors(n_neighbors=10).fit(z_np)
        distances, _ = nbrs.kneighbors(z_np)
        avg_dist = distances[:, 1:].mean(axis=1)
        best_labels = (avg_dist < np.percentile(avg_dist, 70)).astype(int)
        best_score = silhouette_score(z_np, best_labels)

        # 强制保证两类分布（新增）
        if len(np.unique(best_labels)) == 1:
            split_point = len(best_labels) // 2
            best_labels[:split_point] = 1
            best_score = 0.0  # 标记为人工干预

    # ==================== 改进点 4：标签传播稳定性增强 ====================
    try:
        # 动态调整传播参数
        n_neighbors = min(15, len(z_np) // 3)
        label_prop_model = LabelSpreading(
            kernel='knn',
            n_neighbors=n_neighbors,
            alpha=0.2  # 新增平滑参数
        )
        label_prop_model.fit(z_np, best_labels)
        pseudo_labels = label_prop_model.transduction_
        pseudo_probs = label_prop_model.label_distributions_
    except Exception as e:
        print(f"⚠️ 标签传播失败: {str(e)}，直接使用原始标签")
        pseudo_labels = best_labels
        pseudo_probs = np.eye(len(np.unique(pseudo_labels)))[pseudo_labels]

    # ==================== 改进点 5：动态置信度阈值 ====================
    entropy = -np.sum(pseudo_probs * np.log(pseudo_probs + 1e-8), axis=1)

    # 根据聚类质量动态调整阈值
    if best_score < 0.1:
        percentile = 95  # 低质量聚类时放宽阈值
    elif best_score < 0.2:
        percentile = 90
    else:
        percentile = 85

    high_confidence_mask = entropy < np.percentile(entropy, percentile)

    # 强制保证最小样本量（新增）
    if high_confidence_mask.sum() < 20:
        print("⚠️ 高置信样本不足，使用前20%样本")
        high_confidence_mask = entropy < np.percentile(entropy, 80)

    print(f"最终轮廓系数: {best_score:.2f} | 高置信样本: {high_confidence_mask.mean():.1%}")
    return pseudo_labels, pseudo_probs, high_confidence_mask
# 在construct_prior函数中确保返回正确形状
from sklearn.linear_model import Lasso
import numpy as np


def construct_prior(X, pseudo_labels, high_confidence_mask):
    X_high = X[high_confidence_mask]
    y_pseudo = pseudo_labels[high_confidence_mask].astype(float)

    # 第一阶段：试点Lasso获取初始系数 [6](@ref)
    lasso_pilot = Lasso(alpha=0.05, max_iter=5000)  # 降低初始alpha
    lasso_pilot.fit(X_high, y_pseudo)

    # 自适应权重计算：系数倒数 (网页6)
    weights = 1 / (np.abs(lasso_pilot.coef_) + 1e-4)

    # 第二阶段：带权重的Lasso [6,7](@ref)
    lasso = Lasso(alpha=0.1 * np.mean(weights),  # 全局alpha按权重缩放
                  max_iter=10000,
                  tol=1e-5)
    lasso.fit(X_high, y_pseudo)

    # 非零保护机制优化
    if np.all(lasso.coef_ == 0):
        # 基于试点结果生成扰动（替代随机初始化）
        lasso.coef_ = 0.1 * lasso_pilot.coef_ + np.random.normal(scale=0.01, size=X_high.shape[1])

    return lasso.coef_.astype(np.float32), float(lasso.intercept_)





#############################
# 4. 贝叶斯分位数回归（修正：集成动态先验）
#############################
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


def bayesian_quantile_regression(X, y, tau=0.30, prior_coef=None, prior_intercept=None,
                                 n_steps=3000, lr=0.01):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    n_samples, n_features = X.shape

    def model(X, y):
        # 添加特征维度的plate
        with pyro.plate("features", n_features):
            if prior_coef is not None:
                beta_loc = torch.tensor(prior_coef, dtype=torch.float32)
                beta = pyro.sample("beta", dist.Normal(beta_loc, 0.1 * torch.ones(n_features)))
            else:
                beta = pyro.sample("beta", dist.Normal(torch.zeros(n_features), 1.0))

        # 处理截距项
        with pyro.plate("intercept_plate", 1):
            if prior_intercept is not None:
                intercept_loc = torch.tensor([prior_intercept], dtype=torch.float32)
                intercept = pyro.sample("intercept", dist.Normal(intercept_loc, 0.1))
            else:
                intercept = pyro.sample("intercept", dist.Normal(0.0, 1.0))

        # 计算预测值
        mu = torch.matmul(X, beta) + intercept

        # 使用事件维度处理观测值
        with pyro.plate("data", X.shape[0]):
            kappa = tau / (1.0 - tau)
            pyro.sample("obs", dist.AsymmetricLaplace(mu, 0.8, kappa).to_event(1),
                        obs=y)

    def guide(X, y):
        # 定义变分参数
        beta_loc = pyro.param("beta_loc", torch.randn(n_features))
        beta_scale = pyro.param("beta_scale", torch.ones(n_features),
                                constraint=dist.constraints.positive)
        intercept_loc = pyro.param("intercept_loc", torch.tensor([0.0]))
        intercept_scale = pyro.param("intercept_scale", torch.tensor([1.0]),
                                     constraint=dist.constraints.positive)

        # 使用plate匹配模型结构
        with pyro.plate("features", n_features):
            pyro.sample("beta", dist.Normal(beta_loc, beta_scale))

        with pyro.plate("intercept_plate", 1):
            pyro.sample("intercept", dist.Normal(intercept_loc, intercept_scale))

    # 训练过程保持不变...
    # 训练过程保持不变...
    # 训练
    pyro.clear_param_store()
    optimizer = Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses = []
    for step in range(n_steps):
        loss = svi.step(X_tensor, y_tensor)
        losses.append(loss)
        if step % 500 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")

    # 获取后验参数
    posterior = {
        "beta": pyro.param("beta_loc").detach().numpy(),
        "intercept": pyro.param("intercept_loc").item()
    }
    from pyro.infer.mcmc import MCMC, NUTS
    #nuts_kernel = NUTS(model)
    #mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=100)
    #mcmc.run(X_tensor, y_tensor)
   # print(mcmc.diagnostics())
    return posterior, losses




def explain_model(posterior, X_sample, feature_names):
    """SHAP特征重要性分析"""
    import shap

    # 定义可解释模型
    class BayesianQuantileWrapper:
        def __init__(self, posterior):
            self.beta = posterior['beta']
            self.intercept = posterior['intercept']

        def predict(self, X):
            return np.dot(X, self.beta) + self.intercept

        def __call__(self, X):  # 新增可调用方法
            return self.predict(X)

    # 创建解释器
    model = BayesianQuantileWrapper(posterior)

    # 使用前100个样本作为背景数据集
    background = shap.sample(X_sample, 120)

    # 正确传递预测函数
    explainer = shap.KernelExplainer(
        model.predict,  # 直接使用predict方法
        background
    )

    # 计算SHAP值（使用前50个样本解释）
    shap_values = explainer.shap_values(X_sample[:100])

    # 可视化
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_sample[:50], feature_names=feature_names, plot_type='bar')
    plt.title("Feature Importance via SHAP Values")
    plt.tight_layout()
    return shap_values







import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from properscoring import crps_ensemble

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from properscoring import crps_ensemble

def posterior_diagnostics(posterior, X_test, y_test):
    """后验预测检验与CRPS计算"""
    # 参数提取
    beta = posterior['beta']
    intercept = posterior['intercept']
    tau = 0.3
    scale = 0.7

    # 生成预测均值
    y_pred_mean = np.dot(X_test, beta) + intercept

    # 生成Asymmetric Laplace分布样本
    n_samples = 4000
    kappa = tau / (1 - tau)
    samples = np.zeros((X_test.shape[0], n_samples))

    # 基于参数化公式生成样本
    for i in range(X_test.shape[0]):
        loc = y_pred_mean[i]
        exp_pos = np.random.exponential(scale=1 / kappa, size=n_samples)
        exp_neg = np.random.exponential(scale=1 / (1 - tau), size=n_samples)
        samples[i, :] = loc + scale * (exp_pos - exp_neg)

    # 计算CRPS（修正：移除转置操作）
    crps = crps_ensemble(y_test, samples)  # 正确的形状对齐

    # 计算覆盖率
    lower = np.percentile(samples, 2.5, axis=1)
    upper = np.percentile(samples, 97.5, axis=1)
    coverage = np.mean((y_test >= lower) & (y_test <= upper))

    # 计算MSE (Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred_mean)
    print("MSE:", mse)

    # 计算RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    # 计算MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_test, y_pred_mean)
    print("MAE:", mae)

    # 可视化（保持不变）
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred_mean, alpha=0.6, label='Predicted vs True')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
    plt.fill_between(y_test, lower, upper, alpha=0.2, label='95% CI')
    plt.xlabel("True Time")
    plt.ylabel("Predicted Mean Time")
    plt.title(f"Posterior Check (CRPS={np.mean(crps):.3f}, Coverage={coverage:.2%}, MSE={mse:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f})")
    plt.legend()
    plt.grid(True)
    plt.show()

    return crps, coverage, mse, rmse, mae


#############################
# 5. 动态更新与主流程（修正：先验集成）
#############################






def dynamic_update_feature_alignment(source_data, target_data, y_target, device,
                                    num_outer_iter=20, update_epochs=20,
                                    latent_dim=15):
    # 初始化参数
    prior_coef, prior_intercept = None, None
    default_posterior = {
        "beta": np.zeros(source_data.shape[1]),
        "intercept": 0.0,
        "tau": 0.5,
        "scale": 1.0
    }
    posterior = default_posterior.copy()
    X_src = np.zeros((1, source_data.shape[1]))  # 防止未定义错误

    for iter in range(num_outer_iter):
        print(f"\n===== Outer Iteration {iter + 1}/{num_outer_iter} =====")
        try:
            # ================== 特征对齐 ==================
            vae, _ = train_feature_alignment(
                source_data, target_data, device,
                num_epochs=update_epochs, latent_dim=latent_dim
            )

            # ================== 生成隐空间特征 ==================
            with torch.no_grad():
                _, _, _, z_src = vae(source_data)
                _, _, _, z_tgt = vae(target_data)

            # ================== 伪标签生成 ==================
            try:
                pseudo_labels, _, high_conf_mask = generate_pseudolabels(z_src)
            except ValueError as ve:
                print(f"伪标签生成失败: {str(ve)}")
                if iter == 0:
                    pseudo_labels = np.random.randint(0, 2, size=len(z_src))
                    high_conf_mask = np.ones_like(pseudo_labels, dtype=bool)
                else:
                    continue



            # ================== 构建先验 ==================
            X_src = source_data.cpu().numpy()
            try:
                coef, intercept = construct_prior(X_src, pseudo_labels, high_conf_mask)

                prior_decay = 0.9  **  (iter + 1)
                prior_coef = coef * prior_decay + (1 - prior_decay) * posterior['beta']

                prior_intercept = intercept * prior_decay + (1 - prior_decay) * np.random.normal(scale=0.1)
                print(f"Prior Updated (decay={prior_decay:.2f}): coef={prior_coef[:5]}..., intercept={prior_intercept:.4f}")
            except Exception as e:
                print(f"先验构建失败: {str(e)}")
                if prior_coef is not None:
                    prior_decay = 0.9 ** (iter + 1)
                    prior_coef = prior_coef * prior_decay + (1 - prior_decay) * np.random.normal(scale=0.1, size=prior_coef.shape)
                    prior_intercept = prior_intercept * prior_decay + (1 - prior_decay) * np.random.normal(scale=0.1)
                    print(f"使用衰减后的随机先验 (decay={prior_decay:.2f}): coef={prior_coef[:5]}...")

            # ================== 贝叶斯回归 ==================
            try:

                posterior, losses = bayesian_quantile_regression(
                    X=target_data.cpu().numpy(),
                    y=y_target,
                    prior_coef=prior_coef,
                    prior_intercept=prior_intercept,
                    n_steps= 10000                )
                if not isinstance(posterior, dict):
                    raise ValueError("Invalid posterior type")
            except Exception as e:
                print(f"贝叶斯回归异常: {str(e)}, 使用默认后验")
                posterior = default_posterior.copy()
                losses = []

            # ================== 可视化 ==================
            plt.figure(figsize=(10, 4))
            plt.plot(losses[100:])  # 跳过初始波动
            plt.title(f"Iteration {iter + 1} Training Loss")
            plt.xlabel("Step")
            plt.ylabel("ELBO Loss")
            plt.show()

        except Exception as e:
            print(f"外层迭代异常: {str(e)}")
            posterior = posterior if isinstance(posterior, dict) else default_posterior.copy()
            continue

    return posterior












#############################
# 6. 主程序
#############################
def main():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    preprocessor = DataPreprocessor(
        wdbc_path='wdbc.data',
        wpbc_path='wpbc.data'
    )
    preprocessor.load_data()
    src_features, tgt_features, tgt_target = preprocessor.preprocess()
    preprocessor.show_preprocessed_data()

    # 转换为Tensor
    source_tensor = torch.from_numpy(src_features).float().to(device)
    target_tensor = torch.from_numpy(tgt_features).float().to(device)

    # 动态训练流程
    posterior = dynamic_update_feature_alignment(
        source_tensor,
        target_tensor,
        tgt_target,
        device,
        num_outer_iter=2,
        update_epochs=20,
        latent_dim=15
    )
    print(posterior)
    print("\nFinal Posterior Parameters:")
    print(f"Final Posterior 类型: {type(posterior)}")
    print(f"Beta: {posterior['beta'][:5]}...")
    print(f"Intercept: {posterior['intercept']:.4f}")

    X_sample = tgt_features.astype(np.float32)
    X_sample_tensor = torch.from_numpy(X_sample).to(device)
    explain_model(posterior, X_sample, preprocessor.common_features)

    # 后验诊断
    crps, coverage, mse, rmse, mae= posterior_diagnostics(posterior,tgt_features,tgt_target.ravel()  # 确保目标变量为1D数组
    )








if __name__ == "__main__":
    main()
