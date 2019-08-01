import sys
import copy
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt


class HHnets:
    """
    Class to simulate a network of Hodgkin-Huxley neurons.
    """
    # default parameters
    dt = 0.01
    C = 9 * np.pi
    G_na = 1080 * np.pi
    G_k = 324 * np.pi
    G_l = 2.7 * np.pi
    E_na = 115
    E_k = -12
    E_l = 10.6

    # choose the motif
    def set_one(self):
        return 0
    def set_twouni(self):
        E_syn = 60
        tmat = self.st_last - self.tau
        mask = tmat < 0
        tmat[mask] = (self.st_second - self.tau)[mask]
        tmat = np.int_(tmat * 100)
        return np.array([0, self.G_syn * self.alpha_func[tmat[0]] * (E_syn - self.statemat[1, 0])])
    def set_twobi(self):
        E_syn = 60
        tmat = self.st_last - self.tau
        mask = tmat < 0
        tmat[mask] = (self.st_second - self.tau)[mask]
        tmat = np.int_(tmat * 100)
        return np.array([self.G_syn * self.alpha_func[tmat[1]] * (E_syn - self.statemat[0, 0]),
                         self.G_syn * self.alpha_func[tmat[0]] * (E_syn - self.statemat[1, 0])])
    def set_chain(self):
        E_syn = 60
        tmat = self.st_last - self.tau
        mask = tmat < 0
        tmat[mask] = (self.st_second - self.tau)[mask]
        tmat = np.int_(tmat * 100)
        return np.array([0,
                         self.G_syn * self.alpha_func[tmat[0]] * (E_syn - self.statemat[1, 0]),
                         self.G_syn * self.alpha_func[tmat[1]] * (E_syn - self.statemat[2, 0])])
    def set_V(self):
        E_syn = 60
        tmat = self.st_last - self.tau
        mask = tmat < 0
        tmat[mask] = (self.st_second - self.tau)[mask]
        tmat = np.int_(tmat * 100)
        return np.array([self.G_syn * self.alpha_func[tmat[1]] * (E_syn - self.statemat[0, 0]),
                         self.G_syn * self.alpha_func[tmat[0]] * (E_syn - self.statemat[1, 0]) + self.G_syn *
                         self.alpha_func[tmat[2]] * (E_syn - self.statemat[1, 0]),
                         self.G_syn * self.alpha_func[tmat[1]] * (E_syn - self.statemat[2, 0])])
    def set_circular(self):
        E_syn = 60
        tmat = self.st_last - np.transpose(np.array([[self.tau, self.tauprime]]))
        mask = tmat < 0
        tmat[mask] = (self.st_second - np.transpose(np.array([[self.tau, self.tauprime]])))[mask]
        tmat = np.int_(tmat * 100)
        return np.array([(self.G_syn * self.alpha_func[tmat[0, 1]] + self.G_prime_syn * self.alpha_func[tmat[1, 2]]) * (
                    E_syn - self.statemat[0, 0]),
                         (self.G_syn * self.alpha_func[tmat[0, 0]] + self.G_syn * self.alpha_func[tmat[0, 2]]) * (
                                     E_syn - self.statemat[1, 0]),
                         (self.G_syn * self.alpha_func[tmat[0, 1]] + self.G_prime_syn * self.alpha_func[tmat[1, 0]]) * (
                                     E_syn - self.statemat[2, 0])])

    # array consisting alpha_func of synapse values for 50 msec
    tauminus = 4
    tauplus = 0.1
    x = np.linspace(0, 50, int(50 / dt) + 1)
    alpha_func = (np.exp(-x / tauminus) - np.exp(-x / tauplus)) / (tauminus - tauplus)

    # define which signal is to be injected
    frequency = 3
    amplitude = 20
    sinus = amplitude * np.sin(
        2 * np.pi * frequency * 0.001 * np.linspace(0, 1000 / frequency, int((1000 / dt) / frequency + 1)))
    sig = np.load("Convolved_whitenoise.npy")
    Gaussian_white = sig[200000:] - sig[200000:].mean()
    temp = np.zeros(500000)
    temp[200000:300000] = sig[100000:200000] - sig[100000:200000].mean()

    def __init__(self, motif, init = [10, 33, 45], I_ext=300, G_syn=10, G_prime_syn=0, tau=0, tauprime=0):
        """
        Decide the motif and the signal
        """

        # random = np.random.randint(low=0, high=149, size=int(len(init)))
        # self.statemat = np.loadtxt("rangen.txt")[
        #     np.array(random)]  # randomly select initial conditions.The columms are: v,m,n,h

        # constants
        self.I_ext = np.full(len(init), I_ext)
        self.G_syn = G_syn
        self.G_prime_syn = G_prime_syn
        self.tau = tau
        self.tauprime = tauprime

        # dynamics
        self.statemat = np.loadtxt("rangen.txt").take(init, axis=0)
        self.counter = 0  # Counter of current time step
        self.st_last = np.full(len(init), 25)  # current time - time of last spike
        self.st_second = np.full(len(init), 25)  # current time - time of second spike
        self.ascending = np.full(len(init), False)  # boolean to check whether voltages are increasing
        # self.signalnum = 0  # used for perturbation (PRC)
        self.total_I = np.zeros(3)  # sum of all external current (I_ext + I_syn + signal)

        # switch for selecting the motif
        I_syn_switch = {"one": self.set_one,
                        "twouni": self.set_twouni,
                        "twobi": self.set_twobi,
                        "chain": self.set_chain,
                        "V": self.set_V,
                        "circular": self.set_circular,
                        }
        self.I_syn_current = I_syn_switch[motif]()  # get I_syn values depending on the motify
        print("Initialised the network motif: {}".format(motif))

    # auxilary functions for calculation
    def alpha_x(self, vv):
        alpha_m = (25 - vv) / (10 * (np.exp((25 - vv) / 10) - 1))
        alpha_n = (10 - vv) / (100 * (np.exp((10 - vv) / 10) - 1))
        alpha_h = 0.07 * np.exp(-vv / 20)
        return np.transpose(np.array([alpha_m, alpha_n, alpha_h]))
    def beta_x(self, vv):
        beta_m = 4 * np.exp(-vv / 18)
        beta_n = 0.125 * np.exp(-vv / 80)
        beta_h = 1 / (np.exp((30 - vv) / 10) + 1)
        return np.transpose(np.array([beta_m, beta_n, beta_h]))

    # let the neuron evolve for one timestep
    def advance(self):
        """
        Let the neuron evolve in one time step
        """
        # Selecting which signal to use, the code can be surely improved using switch
        #         signal= np.insert(np.zeros(self.statemat.shape[0]-1),0,self.sinus[self.counter%int((1000/self.dt)/self.frequency+1)])
        signal = np.insert(np.zeros(self.statemat.shape[0]-1), -1, self.Gaussian_white[self.counter])
        #         signal=self.temp[self.counter]
        # signal = np.zeros(1)
        #         signal= np.insert(np.zeros(self.statemat.shape[0]-1),0,self.signalnum)

        # updating voltages and three gating variables
        self.total_I = self.I_ext + self.I_syn_current + signal
        # + np.random.uniform(low=-1, high=1, size=self.statemat.shape[0])
        old_V = copy.deepcopy(self.statemat[:, 0])
        self.statemat[:, 0] = self.statemat[:, 0] + self.dt * (
        (-self.G_na * self.statemat[:, 1] ** 3 * self.statemat[:, 3] *
        (self.statemat[:, 0] - self.E_na) - self.G_k * self.statemat[:,2] ** 4 *
         (self.statemat[:, 0] - self.E_k) - self.G_l * (self.statemat[:, 0] - self.E_l) +
         self.total_I) / self.C)
        self.statemat[:, 1:4] = self.statemat[:,1:4] + self.dt * (
                self.alpha_x(old_V) * (1 - self.statemat[:,1:4]) -
                self.beta_x(old_V) * self.statemat[:,1:4])
        self.counter += 1

        # three point local maximum decision and updating st
        fired = np.bitwise_and(self.ascending, self.statemat[:, 0] < old_V)
        self.ascending = np.bitwise_and(old_V < self.statemat[:, 0], self.statemat[:, 0] > 50)
        self.st_second[fired] = self.st_last[fired]
        self.st_second = self.st_second + self.dt
        self.st_last = self.st_last + self.dt
        self.st_last[fired] = 0

        # Select network motif

    # create a txt file with information about different variables at time point
    def record(self, datapath, running_time=5000, cutoff=3000):
        """
        Create a txt file at @datapath recording the voltages for @running_time
        """
        for _ in range(int(cutoff / self.dt)):
            self.advance()
        for _ in range(int(running_time / self.dt) - 1):
            if (self.counter) % int((1 / self.dt) / 10) == 0:  # print every 0.1 msec
                with open("{}.txt".format(datapath), "a") as path:
                    tobeprinted = np.array([self.statemat[:, 0]])
                    np.savetxt(path, tobeprinted, fmt="%.3f")
            self.advance()
        return ("recording finished")

    def spiketrain(self, running_time=5000, cutoff=3000):
        """
        Return spiketrain of neurons, start recording after @cutoff msec
        and run for additonal @running_time msec
        """
        running_time = int(running_time / self.dt)
        cutoff = int(cutoff / self.dt)
        spiketrain = np.zeros((self.statemat.shape[0], running_time))
        for _ in range(cutoff):
            self.advance()
        for _ in range(running_time - 1):
            spiketrain[self.st_last == 0, self.counter - cutoff] = 1
            self.advance()
        return spiketrain

    # calculate firing rate of neurons with Gaussian window moving 10 timestep
    def firingrate(self, running_time=5000, cutoff=3000, std=25, save=False, dir="spiketrain.txt"):
        """
        Return firing rate of neurons smoothed with a Gaussian window which moves 10 timestep each time
        """
        std = int(std / self.dt)
        windowsize = std * 20  # Arbitary size
        window = signal.gaussian(windowsize, std)  # 500msec window by default
        result = np.zeros((self.statemat.shape[0], (int(running_time / self.dt) - windowsize // 2) // 10))

        spiketrain = self.spiketrain(running_time=running_time, cutoff=cutoff)  # Compute spiketrain
        if save==True:
            np.savetxt(dir, spiketrain)

        # advance window ten step at a time
        for index in np.arange(0, windowsize // 20):  # When the window don't fully cover the spiketrain yet
            result[:, index] = np.sum(spiketrain[:, 0:index + windowsize // 2] * window[windowsize // 2 - index:],
                                      axis=1)  # 0~250,250~
        for index in range(windowsize // 20 + 1, (running_time - windowsize // 2) // 10):
            result[:, index] = np.sum(spiketrain[:, index * 10 - windowsize // 2:index * 10 + windowsize // 2] * window,
                                      axis=1)  # 100~600 , 0~500
        return result

    def gen_PRC(self, accuracy=20, running_time=150, cutoff=3000, Pulse_strength=20, Pulse_width=2):
        """
        Return spiketrains resulting from perturbuing a neuron at different phases. Complete method for PRC analysis.
        """

        running_time = int(running_time / self.dt)
        cutoff = int(cutoff / self.dt)
        Pulse_width = int(Pulse_width / self.dt)
        Spiketrain = np.zeros((accuracy, self.statemat.shape[0], running_time))
        Signals = np.zeros((accuracy, running_time))
        Orderedspikes = np.zeros((accuracy, self.statemat.shape[0], 5))
        Total_current = np.zeros(((accuracy, self.statemat.shape[0], running_time)))
        # Autonomous activities, let the synchrony be established
        for _ in range(cutoff - 50000):
            self.advance()
        # take #spikes for last 500ms and compute the period without perturbation
        originalspikes = np.zeros(50000)
        for _ in range(cutoff - 50000, cutoff):
            if self.st_last[0] == 0:
                originalspikes[self.counter - (cutoff - 50000)] = 1
            self.advance()
        orgperiod = np.mean(np.diff(np.where((originalspikes))))
        # Save the state of synchronized state and reset for each simulations with pulse at different phases
        syncedstate = np.copy(self.statemat)
        sync_st_last = np.copy(self.st_last)
        sync_st_second = np.copy(self.st_second)
        sync_ascending = np.copy(self.ascending)

        # perturbe at different phases
        for accindex in range(accuracy):
            self.statemat = np.copy(syncedstate)
            self.st_last = sync_st_last
            self.st_second = sync_st_second
            self.ascending = sync_ascending
            self.counter = cutoff
            # find the first spike of neuron 0 after cutoff
            i = 0
            while self.st_last[0] != 0:
                Spiketrain[accindex, self.st_last == 0, self.counter - cutoff] = 1
                Total_current[accindex, :, i] = self.total_I
                i += 1
                self.advance()
            n0firstst = self.counter
            # Advance to the phase of current injection
            while self.counter - (n0firstst + orgperiod) < int(
                    (accindex / (accuracy - 1)) * orgperiod) - Pulse_width / 2:
                Spiketrain[accindex, self.st_last == 0, self.counter - cutoff] = 1
                Total_current[accindex, :, i] = self.total_I
                self.advance()
                i += 1
            # Inject a signal for @Pulse_width timestep
            self.signalnum = Pulse_strength
            for _ in range(Pulse_width):
                Spiketrain[accindex, self.st_last == 0, self.counter - cutoff] = 1
                Signals[accindex, self.counter - cutoff] = self.signalnum
                Total_current[accindex, :, i] = self.total_I
                i += 1
                self.advance()
            # End signal injection,run for @running_time more
            self.signalnum = 0
            while self.counter < running_time + cutoff:
                Spiketrain[accindex, self.st_last == 0, self.counter - cutoff] = 1
                Total_current[accindex, :, i] = self.total_I
                i += 1
                self.advance()
                # Save timepoint of spikes after the perturbation has been reached
            Orderedspikes[accindex, 0] = np.where(Spiketrain[accindex, 0, :])[0][
                                         1:6]  # we don't perturbe after 0th but 1th spike
            n1f = np.where(Spiketrain[accindex, 1, :])[0]
            Orderedspsikes[accindex, 1] = n1f[n1f > Orderedspikes[accindex, 0, 0] + int(self.tau / self.dt)][0:5]
            n2f = np.where(Spiketrain[accindex, 2, :])[0]
            Orderedspikes[accindex, 2] = n2f[n2f > Orderedspikes[accindex, 1, 0] + int(self.tau / self.dt)][0:5]
        print("spiketrain, orderedspikes, signals, orgperiod, total current")
        return (Spiketrain, Orderedspikes, Signals, orgperiod, Total_current)

    def F_I(self, I_ext_interval, running_time=5000, std=25):
        """
        Return firing rates of spikes by injecting different I_exts defined by @I_ext_interval
        The I_ext will first monotonically increase to max and decrease to min to check bistability
        """
        std = int(std / self.dt)
        windowsize = std * 20
        window = signal.gaussian(windowsize, std)
        running_time = int(running_time / self.dt)
        spiketrain = np.zeros((3, running_time))  # 3,300000
        result = np.zeros((3, (running_time - windowsize // 2) // 10))  # 3,275000
        I_exts = np.concatenate(
            (np.linspace(250, 350, running_time // 2), np.flip(np.linspace(250, 350, running_time // 2), axis=0)),
            axis=0)
        for _ in range(running_time - 1):
            self.I_ext = I_exts[_]
            self.advance()
            spiketrain[self.st_last == 0, self.counter] = 1
        for index in np.arange(0, windowsize // 20):
            result[:, index] = np.sum(spiketrain[:, 0:index + windowsize // 2] * window[windowsize // 2 - index:],
                                      axis=1)  # 0~250,250~
        for index in range(windowsize // 20 + 1, (running_time - windowsize // 2) // 10):
            result[:, index] = np.sum(spiketrain[:, index * 10 - windowsize // 2:index * 10 + windowsize // 2] * window,
                                      axis=1)  # 100~600 , 0~500
        return result



# I.Record voltage activities
cnet = HHnets(motif="V", tau=10)
cnet.record(datapath="tau=10.txt")

# II.Compute firing rate and save the spike train with signal modulation
np.save("datas/firingrate/tau=10.npy",
        cnet.firingrate(save=True,dir="datas/spiketrain/tau=10.txt") )