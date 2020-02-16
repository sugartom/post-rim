import matplotlib.pyplot as plt

batch_size = [1, 2, 4, 8, 16, 32, 64, 128]
runtime = [0.055765, 0.121874, 0.354351, 0.377689, 0.683596, 1.353627, 2.601987, 5.169096]

plt.plot(batch_size, runtime, '-*', label = 'ssd_resnet')
# plt.xscale('log')
plt.xlabel('batch size')
plt.ylabel('runtime (sec)')
plt.legend()
plt.show()