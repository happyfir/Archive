# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(7, 5), dpi=180)

size = 5
x = np.arange(size)

before_f1 = [0.8199, 0.9074, 0.9063, 0.9153, 0.9263]
after_f1 = [0.8011, 0.8662, 0.8884, 0.8718, 0.9111]

before_acc = [0.8666, 0.9305, 0.9292, 0.9352, 0.9449]
after_acc = [0.8499, 0.8917, 0.9150, 0.8973, 0.9344]

before_Recall = [0.8289,0.9047, 0.8976,0.8954, 0.9275]
after_Recall = [0.7946, 0.8055, 0.8738, 0.8181, 0.9245]

before_AUC = [0.9334, 0.9797, 0.9774, 0.9833, 0.9843]
after_AUC = [0.9205, 0.9665, 0.9516, 0.9627, 0.9681]

plt.ylim(0.75, 1.0)
total_width, n = 0.8, 5
width = total_width / n
x = x - (total_width - width) / 2

plt.plot(x,before_Recall, marker = 'o', markersize = 4)
plt.plot(x,after_Recall, marker = 'o', markersize = 4)

# 在左侧显示图例
plt.legend(['original','intervention'])
for a,b in zip(x,before_Recall):
    plt.text(a, b , b ,ha = 'center',va = 'bottom', fontsize = 10)
for a,b in zip(x,after_Recall):
    plt.text(a, b , b ,ha = 'center',va = 'top', fontsize = 10)

# 设置标题
# plt.title("Comparison of Precision and False Alarm Rate\n When Using Different Feature Schemes")
# 为两条坐标轴设置名称
plt.ylabel("Compare Recall")

aaa = ['Xing', 'Agrawal', 'Catak', 'Zhang','Mal-CLAM']
bbb = range(5)
plt.xticks(bbb, aaa)

plt.savefig("Abation/robust_Recall.jpg")

plt.show()