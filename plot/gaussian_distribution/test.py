import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.mlab import normpdf
from scipy.stats import norm


# section = np.arange(-1, 1, 1/20.)
# plt.fill_between(section,normpdf(section, 0, 1))
# plt.show()


x = np.arange(0, 10, 1/10.)
y = norm.pdf(x, 5, 1.5)

plt.plot(x, y, 'k-', linewidth = 3)

# x1 = 6.9
# y1 = 0
# x2 = x1
# y2 = y[int(x1*10)]
# plt.plot([x1, x2], [y1, y2], 'k-', linewidth = 3)

x_fill = np.arange(0, 7, 1/10.)
y_fill = y[:len(x_fill)]
plt.fill_between(x_fill, y_fill, color = 'cornflowerblue')

# plt.text(7, 0.15, "n = n*", fontsize = 36)
# plt.arrow(7.5, 0.13, -0.6, -0.13)
plt.annotate("n = n*", xy = (6.9, 0), xytext = (7, 0.15), arrowprops=dict(arrowstyle="->", linewidth = 3), fontsize = 36)

plt.xlabel("n", fontsize = 36)
plt.ylabel("f(n)", fontsize = 36)

axes = plt.gca()
plt.setp(axes.get_xticklabels(), visible = False)
plt.setp(axes.get_yticklabels(), visible = False)
axes.set_ylim([0, 0.3])
axes.set_xlim([0, 10])

plt.subplots_adjust(top=0.95, bottom=0.12, left=0.08, right=0.95, hspace=0.20, wspace=0.35)

fig = plt.gcf()
fig.set_size_inches(15, 6)

# plt.show()
plt.savefig("gaussian_distribution.pdf")