import numpy as np
import matplotlib.pyplot as plt

n_files = 100
path = './experiments/test9/PES'
name = '/test9_PES_f_'
n_iterations = 100
log_regret = np.zeros((n_iterations,n_files))
time  = np.zeros((n_iterations,n_files))
real_opt = 0.#test7:4.389940124468381 #test5:-0.5369910241891562#test-0.42973174#test_linear2:1.382850185589923#test_linear4:0.90579135#test_linear8:0.84455888
for i in range(n_files):
    print(i)
    #path = './experiments/test9/PES'
    #name = '/test9_'
    log_regret[:, i] = np.log10(np.abs(real_opt - np.loadtxt(path + name+str(i)+'.txt', unpack=True)))[0:n_iterations]
    #time[:, i] = np.loadtxt(path + name + str(i) + '.txt', unpack=True)[1, 0:n_iterations]/60

log_regret_stats = np.zeros((n_iterations, 2))
log_regret_stats[:, 0] = np.mean(log_regret, axis=1)
log_regret_stats[:, 1] = np.std(log_regret, axis=1)

#time_stats = np.zeros((n_iterations, 2))
#time_stats[:, 0] = np.mean(time, axis=1)
#time_stats[:, 1] = np.std(time, axis=1)
    
np.savetxt(path + name + 'log_regret_stats.txt', log_regret_stats)
#np.savetxt(path + name + '_time_stats.txt', time_stats)
plt.figure()
plt.plot(log_regret_stats[:, 0], label=name)
plt.plot(log_regret_stats[:, 0] + log_regret_stats[:, 1])
plt.plot(log_regret_stats[:, 0] - log_regret_stats[:, 1])
#plt.figure()
#plt.plot(time_stats[:, 0], label=name)
#plt.plot(time_stats[:, 0] + time_stats[:, 1])
#plt.plot(time_stats[:, 0] - time_stats[:, 1])
plt.show()