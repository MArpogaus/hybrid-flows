# %% imports
import numpy as np
import pandas as pd
import pyvinecopulib as pv
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow_probability import distributions as tfd

# %% Load Data
df = pd.read_csv("copula.csv")

u_X = np.stack([df["u_x1"], df["u_x2"]], 1)

# %% Plot Original Data
sns.jointplot(df, x="x1", y="x2")
plt.suptitle("Original Data")
plt.tight_layout()

# %% Plot Z Tranformed Data
# This is the data normalized by the elementwise flow (T_1 in our paper)
sns.jointplot(df, x="z_x1", y="z_x2")
plt.suptitle("Z Transformed Data")
plt.tight_layout()

# %% Plot U Tranformed Data
# This is the data transformed by the cdf of the elementwise flow (T_1 in our paper)
sns.jointplot(df, x="u_x1", y="u_x2")
plt.suptitle("U Transformed Data")
plt.tight_layout()

# %% Fit Bivariate Copula model
bicop = pv.Bicop(u_X)
bicop

# %% Plot Copula
n = 200
x = np.linspace(0, 1, n)
mesh = np.meshgrid(x, x)
grid = np.array(mesh).reshape(2, -1)

pdf = bicop.pdf(grid.T)

plt.contour(*mesh, pdf.reshape(n, -1))
plt.scatter(df["u_x1"], df["u_x2"])
plt.suptitle("Copula")
plt.tight_layout()

# %% Transform to Uniform
# Rosenblatt Trafo
# https://vinecopulib.github.io/rvinecopulib/reference/rosenblatt.html
x_uni = bicop.hfunc1(u_X)
sns.jointplot(x=df.u_x1, y=x_uni)

# %% Transform to Normal
x_norm = tfd.Normal(0, 1).quantile([df.u_x1, x_uni]).numpy().T
sns.jointplot(x=x_norm[..., 0], y=x_norm[..., 1])

# %%
# wenn basis unabhänig gleichverteilt -> Basis Dichte == 1
# p_y = p_z2(h(y))*|det \nabla h(y)|
# c(y) = p_y(y) / p_z1(y)
