import matplotlib.pyplot as plt

batch_size = [1, 2, 4, 8, 16, 32, 64, 128]
# runtime = [0.055765, 0.121874, 0.354351, 0.377689, 0.683596, 1.353627, 2.601987, 5.169096]

runtime = dict()
runtime["ssd_inception"] = [0.030863, 0.046375, 0.079598, 0.172221, 0.369045, 0.769243, 1.446811, 2.820706]
runtime["ssd_resnet"] = [0.053909, 0.108926, 0.343045, 0.377029, 0.702654, 1.345150, 2.602157, 5.222773]
runtime["ssd_mobilenet"] = [0.026441, 0.041863, 0.076850, 0.169867, 0.377116, 0.747565, 1.486375, 2.901526]

runtime["rcnn_inception"] = [0.048296, 0.075907, 0.359640, 0.622929, 0.726526, 1.257938, 2.441359]
runtime["rcnn_resnet50"] = [0.087893, 0.230288, 0.486781, 0.606455, 1.000942, 1.838731]
runtime["rcnn_resnet101"] = [0.095433, 0.247921, 0.485277, 0.617229, 1.153166, 2.181561]

throughput = dict()

for k, v in runtime.iteritems():
  throughput[k] = []
  for i in range(len(v)):
    throughput[k].append(batch_size[i] / v[i])

# for k, v in runtime.iteritems():
#   plt.plot(batch_size[:len(v)], v, '-*', label = k)

for k, v in throughput.iteritems():
  plt.plot(batch_size[:len(v)], v, '-*', label = k)

axes = plt.gca()
# axes.set_xlim([xmin,xmax])
axes.set_ylim([0, 70])

plt.xscale('log')
plt.xlabel('batch size')
plt.ylabel('runtime (sec)')
plt.legend(loc = 1)
plt.show()