# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(7, 5), dpi=180)

size = 4
x = np.arange(size)
#
# a = [0.8666, 0.8289, 0.8199, 0.9344]
# b = [0.9305, 0.9047, 0.9074, 0.9797]
# c = [0.9292, 0.8976, 0.9063, 0.9774]
# d = [0.9352, 0.8954, 0.9153, 0.9833]
# e = [0.9449, 0.9275, 0.9263, 0.9843]
#
a = [0.8499, 0.7946, 0.8011, 0.9205]
b = [0.8917, 0.8055, 0.8662, 0.9665]
c = [0.9150, 0.8738, 0.8884, 0.9516]
d = [0.8973, 0.8181, 0.8718, 0.9627]
e = [0.9344, 0.9245, 0.9111, 0.9681]


plt.ylim(0.75, 1.0)
total_width, n = 0.8, 5
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a, width=width, label='Xing')
plt.bar(x + width, b, width=width, label='Agrawal')
plt.bar(x + width + width, c, width=width, label='Catak')
plt.bar(x + width + width + width, d, width=width, label='Zhang')
plt.bar(x + width + width + width + width, e, width=width, label='Mal-CLAM')
# 在左侧显示图例
plt.legend(loc="upper left",prop = {'size':8})

# 设置标题
# plt.title("Comparison of Precision and False Alarm Rate\n When Using Different Feature Schemes")
# 为两条坐标轴设置名称
plt.ylabel("模型检测结果对比")

aaa = ['ACC', 'Recall', 'F1-score', 'AUC']
bbb = range(4)
plt.xticks(bbb, aaa)

plt.savefig("Abation/result1.jpg")

plt.show()