import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from IPython.display import clear_output

class Network:
    def __init__(self):
        self.num_layers = 0
        self.layers = []
        self.dt = .1
    
    def add(self, layer):
        '''adds layer to list of layers and increases number of layers by 1'''
        self.layers.append(layer)
        self.num_layers += 1
        
        
    def run(self, time):
        ''' run the network for a period of time (seconds). this function also stores the voltage/threshold data for each layer'''
        num_steps = int(time / self.dt)
        
        # set up arrays to store values, because we cant set up without knowing the time
        for layer in self.layers:
            layer.Vs = np.zeros((num_steps, layer.num)) # holds voltage
            layer.Ths = np.zeros((num_steps, layer.num)) # holds threshold 

        for i in range(num_steps):
            for layer in self.layers:
                voltage, threshold = layer.neurons.step() #TODO: should this be in layer or not, after connection? should there be another model thing
                layer.Vs[i, :] = voltage.reshape(layer.num,)
                layer.Ths[i, :] = threshold.reshape(layer.num,)             

        
            
    def plot(self):
        for layer in self.layers:
            layer.plot()
        # some kinda plotting


class Layer:


    def __init__(self, name, num, neurontype=None, dt = .1):
        self.num = num
        self.name = name
        self.dt = dt

        
        if neurontype == None:
            self.neurontype = "LIFNeuron"
            self.neurons = LIFNeuron(self.name, self.num)
        elif neurontype == "LIFNeuron":
            self.neurontype = "LIFNeuron"
            self.neurons = LIFNeuron(self.name, self.num)
            ## put more neurontypes here as we make them
        else:
            print("undefined type")
            
    def step(self):
        self.neurontype.step(self) #TODO: this doesnt work
        
        
        
    def __str__(self):
        return f"a layer containing {self.num} neurons"
    
    
    def connection(self, higherlayer):
        pass
        
    # goes through neurons in neurontype and makes arrays holding each info
    def plot_wave(self):
        self.neurontype.plot_wave(self) #TODO: this doesnt work
    

    def plot(self):
        '''plots the threshold, voltage, and spikes of the layer'''
        # sample is the number of neurons to show (in case u have a lot and want to see more clearly)
        num_steps = self.Vs.shape[0]
        time = num_steps * self.dt
        
        plt.figure(figsize=(20, 6))
        plt.subplot(131)

        plt.plot(np.arange(time, step=self.dt), self.Vs[:,:])
        
        #plt.plot(sim.trange(), sim.data[threshold_probe], label = 'threshold')
        plt.xlabel('time [s]')
        plt.ylabel('voltage')
        #plt.legend()
    
        
        plt.subplot(132)
        plt.plot(np.arange(time, step=self.dt), self.Ths[:,:])
        plt.xlabel('time [s]')
        plt.ylabel('threshold')
        
        # TODO: add spikes
        
        
# should take in number of neurons, neurontype, activation? 
# connecting layers should take in inital weights and learning rule? or just learnign rule, and weights are initiated


class LIFNeuron:
# need to make neuron types that can be put into the layer object/class
# make a general neurontype class and then make subclasses that are children?
# makes arrays for voltage and stuff
    dt = .1

# make normal version, then extend for L1 and L2 specific
    def __init__(self, name, num,  ai = 16, ao = 3, ri=2, ro=5 , sqR = 28):
        # attributes that take arguments
        self.name = name
        self.num = num
        self.ai = ai
        self.ao=ao
        self.ri=ri
        self.ro=ro
        self.sqR = sqR
        
        self.nx = self.sqR * np.random.rand(num, 2)
        D = squareform(pdist(self.nx))
        
        # attributes that don't take in arguments
        
        self.S = adjacency(D, ai, ao, ri, ro)
        self.voltage = np.zeros((self.num, 1))
        self.threshold = np.ones((self.num, 1))
        self.spiked = np.zeros((self.num, 1))
        self.v_reset = .1 * np.random.randn(self.num, 1)**2
        
        self.counter = 0
        
        ## variables for storing voltage and threshold over time


        #self.min_voltage = min_voltage
        #self.S_x = S_x
        #self.wave = wave
        # self.thresh = thresh
    
    def step(self, x=None):
        '''updates the voltage and threshold value of the neuron using previous values and possible inputs from other layers. useful so that we can use the step of any neuron type in the layer'''
        tau_v = 2
        tau_u = .3
        tau_th = 60; th_plus = 9; v_th = 1;
        dt = LIFNeuron.dt
        
        
        eta = 3*np.random.randn(self.num, 1)
        y = np.zeros((self.num,1 ))
         # LGN competition rule
        if x != None:
            y = np.maximum(x - self.thresh, 0) # ReLU activation function 
            win, maxInd = np.max(y), np.argmax(y)
            y[y < win] = 0 # only allow winner to participate (make all other entries 0)

        U = np.matmul(self.S,self.spiked) + y # np.matmul(self.S_x, y)
        dV = 1/tau_v * (-1* self.voltage + U + eta)
        self.voltage[:] += dV * dt


                
        # step threshold voltage (theta)
        dTh = (1/tau_th * (v_th - self.threshold) * (1-self.spiked) + th_plus * self.spiked) 

        self.threshold[:] += dTh * dt

        
        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = self.voltage > self.threshold
        self.spiked[:] = spiked_mask * (1)



        # set spiked voltages to v_reset, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        self.voltage[self.voltage < 0] = 0 # min_voltage 
        self.voltage[spiked_mask] = self.v_reset[spiked_mask]
        
        self.counter += 1
        return self.voltage, self.threshold
        
        
        
    def plot_wave(self):

        # Visualization of Wave
        plt.scatter(self.nx[:,0], self.nx[:,1], color = 'k')
        plt.title(self.name + 't =' + str(self.counter))
        fired = np.argwhere(self.spiked)
        plt.scatter(self.nx[fired,0], self.nx[fired,1], color = 'r')
        
        plt.show()
        clear_output(wait=True)
        
    
    

 #class learningrule:
# be able to change learning rule in connection

#class connection:
# be able to initiate weights if dont exist yet, and then uses the learningrule to update



## FUNCTIONS
# todo: decide where to keep

def adjacency(D, ai, ao, ri, ro):
    """Calculate intra-layer adjacency matrix encoding the spatial connectivity 
    of neurons within the layer.

    Parameters
    ----------
    D  : distance matrix
    ai : inhibition amplitude factor
    a0 : excittation amplitude factor
    ri : float
        Excitation radius
    ro : float
        Inhibition radius
    """
    
    S = ai * (D < ri) - (ao * (D > ro) *  np.exp(-D/10))  
    S = S - np.diag(np.diag(S)) 

    return S
