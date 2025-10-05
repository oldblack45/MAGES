import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 构建网格
G11u = np.linspace(180, 320, 100)
G12u = np.linspace(180, 320, 100)
G11u, G12u = np.meshgrid(G11u, G12u)

# 假设函数
Up = G11u * G12u / 100
Ut = (G11u - 200)**2 + (G12u - 200)**2
U11 = G11u**1.2 - G12u
U12 = G12u**1.2 - G11u

fig = plt.figure(figsize=(10, 8))

def plot_subplot(ax, X, Y, Z, zlabel, title):
    surf = ax.plot_surface(
        X, Y, Z,
        cmap='coolwarm',       # 柔和 colormap
        edgecolor='none',     # 去掉边框线
        alpha=0.9,            # 半透明
        linewidth=0,
        antialiased=True
    )
    ax.set_xlabel(r'$G_{u}^{11}$', fontsize=10)
    ax.set_ylabel(r'$G_{u}^{12}$', fontsize=10)
    ax.set_zlabel(zlabel, fontsize=10)
    ax.set_title(title, fontsize=12)

# 四个子图
ax1 = fig.add_subplot(221, projection='3d')
plot_subplot(ax1, G11u, G12u, Up, r'$U_p$', '(a) Utility of the provider')

ax2 = fig.add_subplot(222, projection='3d')
plot_subplot(ax2, G11u, G12u, Ut, r'$U_t^1$', '(b) Utility of the tenant')

ax3 = fig.add_subplot(223, projection='3d')
plot_subplot(ax3, G11u, G12u, U11, r'$U_u^{11}$', '(c) $U^{11}_u$')

ax4 = fig.add_subplot(224, projection='3d')
plot_subplot(ax4, G11u, G12u, U12, r'$U_u^{12}$', '(d) $U^{12}_u$')

plt.tight_layout()
plt.show()
