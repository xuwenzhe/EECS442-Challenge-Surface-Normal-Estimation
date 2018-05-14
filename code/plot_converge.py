import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('trainLog.txt')


plt.figure()
plt.plot(data[:,0],data[:,1])
plt.xlabel('Iteration')
plt.ylabel('Loss0 + Loss1 + Loss2')
plt.show()


# linex = np.arange(20)
# liney = np.zeros(20)
# for i in range(20):
# 	tmpdata = data[i*100: (i+1)*100,1]
# 	tmpvalue = np.mean(tmpdata)
# 	liney[i] = tmpvalue
# 	print liney[i]

# print data[:,1].min()
# plt.figure()
# # plt.plot(data[:,0],data[:,1])
# plt.plot(linex,liney)
# plt.show()