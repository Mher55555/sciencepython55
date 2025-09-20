import numpy as np
import matplotlib.pyplot as plt

# ===== Параметры =====
# Сфера
center = np.array([0.0, 0.0, 1.0])  # центр сферы
r = 1.2                             # радиус

# Конус (ось Oz, вершина в (0,0,0))
alpha = np.pi/6   # угол (30 градусов)
tan_a = np.tan(alpha)

# ===== Сетка =====
N = 200
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
X, Y = np.meshgrid(x, y)
Z = np.linspace(-2, 2, N)

# ===== Уравнение сферы =====
def sphere_eq(x, y, z):
    return (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 - r**2

# ===== Уравнение конуса =====
def cone_eq(x, y, z):
    return z**2 - (x**2+y**2) * tan_a**2

# ===== 3D Визуализация =====
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')

# --- Сфера (поверхность)
phi = np.linspace(0, np.pi, 50)
theta = np.linspace(0, 2*np.pi, 50)
phi, theta = np.meshgrid(phi, theta)
Xs = center[0] + r * np.sin(phi) * np.cos(theta)
Ys = center[1] + r * np.sin(phi) * np.sin(theta)
Zs = center[2] + r * np.cos(phi)
ax.plot_surface(Xs, Ys, Zs, alpha=0.3, color='b')

# --- Конус
h = 2
t = np.linspace(0, h, 50)
ang = np.linspace(0, 2*np.pi, 50)
T, ANG = np.meshgrid(t, ang)
Xc = T * np.tan(alpha) * np.cos(ANG)
Yc = T * np.tan(alpha) * np.sin(ANG)
Zc = T
ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='orange')

# --- Пересечение (численно ищем точки)
pts = []
for _ in range(5000):
    xx, yy = np.random.uniform(-2, 2), np.random.uniform(-2, 2)
    zz = np.random.uniform(0, 2)
    if abs(sphere_eq(xx, yy, zz)) < 0.05 and abs(cone_eq(xx, yy, zz)) < 0.05:
        pts.append([xx, yy, zz])

pts = np.array(pts)
if pts.size > 0:
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=5, c='r')

ax.set_title("Сфера и конус — 3D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# ===== 2D-сечение (плоскость X=0) =====
ax2 = fig.add_subplot(122)
Yline = np.linspace(-2,2,400)
Zline = np.linspace(-2,2,400)
YY, ZZ = np.meshgrid(Yline, Zline)

# Сфера в сечении X=0
F_s = (0-center[0])**2 + (YY-center[1])**2 + (ZZ-center[2])**2 - r**2
ax2.contour(YY, ZZ, F_s, levels=[0], colors='b')

# Конус в сечении X=0
F_c = ZZ**2 - (YY**2) * tan_a**2
ax2.contour(YY, ZZ, F_c, levels=[0], colors='orange')

ax2.set_title("Сечение X=0 (2D)")
ax2.set_xlabel("Y")
ax2.set_ylabel("Z")
ax2.set_aspect('equal')
ax2.grid(True)

plt.tight_layout()
plt.show()