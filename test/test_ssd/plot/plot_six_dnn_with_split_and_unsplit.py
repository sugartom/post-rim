import matplotlib.pyplot as plt

batch_size = [1, 2, 4, 8, 16, 32]

runtime = dict()

# runtime["ssd_inception"] = dict()
# runtime["ssd_inception"]["unsplit"] = [0.030399, 0.045893, 0.090424, 0.157364, 0.378169, 0.741952]
# runtime["ssd_inception"]["split"] = [0.056716, 0.08686, 0.159582, 0.311195, 0.592968]

runtime["ssd_resnet"] = dict()
runtime["ssd_resnet"]["unsplit"] = [0.053528, 0.134642, 0.409647, 0.392608, 0.707587, 1.35695]
runtime["ssd_resnet"]["split"] = [0.14281, 0.3442, 0.501901, 0.632237, 1.17366]

# runtime["ssd_mobilenet"] = dict()
# runtime["ssd_mobilenet"]["unsplit"] = [0.02589, 0.041504, 0.085726, 0.154732, 0.381151, 0.749428]
# runtime["ssd_mobilenet"]["split"] = [0.043822, 0.077283, 0.146872, 0.286088, 0.579157]

# runtime["rcnn_inception"] = dict()
# runtime["rcnn_inception"]["unsplit"] = [0.046934, 0.081453, 0.245074, 0.666709, 0.704551, 1.279884]
# runtime["rcnn_inception"]["split"] = [0.075293, 0.18745, 0.244862, 0.671323, 1.152136]

# runtime["rcnn_resnet50"] = dict()
# runtime["rcnn_resnet50"]["unsplit"] = [0.092471, 0.271941, 0.469224, 0.876678, 0.991751, 1.87032]
# runtime["rcnn_resnet50"]["split"] = [0.224281, 0.415887, 0.765681, 0.774953, 1.624845]

# runtime["rcnn_resnet101"] = dict()
# runtime["rcnn_resnet101"]["unsplit"] = [0.100148, 0.259781, 0.481014, 0.742683, 1.149634, 2.198862]
# runtime["rcnn_resnet101"]["split"] = [0.222746, 0.505742, 0.489422, 0.938435, 1.990255]

throughput = dict()

for k in runtime.keys():
  throughput[k] = dict()
  throughput[k]["unsplit"] = []
  for i in range(6): # batch from 1 to 32
    throughput[k]["unsplit"].append(batch_size[i] / runtime[k]["unsplit"][i])

  throughput[k]["split"] = []
  for i in range(5): # batch from 2 to 32
    throughput[k]["split"].append(batch_size[i + 1] / runtime[k]["split"][i])



# latency
plt.subplot(1, 2, 1)
for k in runtime.keys():
  plt.plot(batch_size, runtime[k]["unsplit"], '-o', label = "%s-unsplit" % k)
  plt.plot(batch_size[1:], runtime[k]["split"], '-*', label = "%s-split" % k)

plt.legend()
plt.xscale('log')
plt.xlabel("batch size")
plt.ylabel("runtime (sec)")

# throughput
plt.subplot(1, 2, 2)
for k in throughput.keys():
  plt.plot(batch_size, throughput[k]["unsplit"], '-o', label = "%s-unsplit" % k)
  plt.plot(batch_size[1:], throughput[k]["split"], '-*', label = "%s-split" % k)

axes = plt.gca()
axes.set_ylim([0, 70])

plt.legend()
plt.xscale('log')
plt.xlabel("batch size")
plt.ylabel("throughput (image/sec)")

plt.show()