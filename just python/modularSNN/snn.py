import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from IPython.display import clear_output

###################################################################################################
################################ A Module for Multi-layered SNNs ##################################                               
######################################## Nora Griffith ############################################
######################################## September 2020 ###########################################
###################################################################################################

class Network:
    """
    A class for a network object that holds all layers
    
    attributes
    -----------
    num_layers (int): number of layers in the network
    layers          : list of layers
    weights         : list of weights in between layers
    learning_rules  : list of learning rules corresponding to the weights list 
    """
        
    def __init__(self):
        """Returns an empty network"""
        self.num_layers = 0
        self.layers = []
        self.weights = []
        self.learning_rules = []
        
        
    def __str__(self):
        return f"a layer containing {self.num} neurons"
    
    
    def add(self, layer, learningrule=None, mu=2.5, sigma=.14):
        """
        Adds layer to list of layers, creates a weight matrix with specified learning rule if the
        added layer is not the first layer, and increases number of layers by 1
        
        parameters
        -----------
        layer        : layer object to add to network
        learningrule : learning rule object to modify the weights between the added layer and the previous layer
        mu           : mean for normally distributed initial weights
        sigma        : std dev for normally distributed initial weights
        """
        
        self.num_layers += 1
        if len(self.layers) > 0:
            pre_num = self.layers[-1].num
            W = np.random.normal(mu, sigma, (pre_num, layer.num)) # shape is (pre, post)
            W = W/np.mean(W, axis = 0) * mu
            self.weights.append(W)
            self.layers.append(layer)
            self.learning_rules.append(learningrule)

        else:
            self.layers.append(layer)
        
    def run(self, time, dt = .1):
        """
        Runs the network. creates arrays to store voltage, threshold, and position of spiked
        neurons and stores this data for each layer at each time step
        
        parameters
        ----------
        time : length of time to run the network for (seconds)
        dt   : length of time step
        """
        
        num_steps = int(time / dt)
        
        # set up arrays to store values, because we cant set up without knowing the time
        for layer in self.layers:
            layer.dt = dt # set the dt here as attribute of layer class so that we can plot individual layers later
            layer.Vs = np.zeros((num_steps, layer.num)) # holds voltage
            layer.Ths = np.zeros((num_steps, layer.num)) # holds threshold 
            layer.Sp = np.zeros((num_steps, layer.num))

        for i in range(num_steps):
            for j, layer in enumerate(self.layers):
                if j > 0:
                    self.weights[j-1] = self.learning_rules[j-1].update(self.weights[j-1], self.layers[j-1], layer, dt = dt) 
                    
                # step function changes the values of the current layer, so here we store those values for each step
                if j == 0: # if the first layer
                    layer.neurons.step_first(dt) #TODO: should this be in layer or not, after connection? should there be another model thing
                else:
                    layer.neurons.step_second(J, dt)
                layer.Vs[i, :] = layer.neurons.voltage.reshape(layer.num,)
                layer.Sp[i, :] = layer.neurons.spiked.reshape(layer.num,)
                layer.Ths[i, :] = layer.neurons.threshold.reshape(layer.num,)
                
                if j < self.num_layers - 1: # as long as its not the last layer

                    J = np.matmul(np.transpose(self.weights[j]), layer.neurons.spiked) # input from current layer to next layer

            
    def plot(self, num):
        """Plots the voltage and threshold versus time plots of all layers in the network
        parameters
        ----------
        num : specifies how many neurons to plot. randomly plots the values of that many neurons,
              otherwise plots all neurons in the layer
        """
        
        for layer in self.layers:
            layer.plot(num)


class Layer:
    """
    A class to represent a generic layer of neurons
    
    attributes
    ----------
    name       : name of the layer, for plotting purposes
    num        : number of neurons in the layer
    neurons    : neurontype object representing neurons in the layer
    dt         : time step (sec), specified in the network's run function, for plotting purposes
    Vs         : array storing voltage values of each neuron in the layer for each time step
    Sp         : array storing spiked values of each neuron in the layer for each time step
    Ths        : array storing threshold values of each neuron in the layer for each time step
    """
    
    def __init__(self, name, num, neurontype=None):
        """
        Returns a layer object
        
        parameters
        ----------
        name       : name of the layer, for plotting purposes
        num        : number of neurons in the layer
        neurontype : neurontype object that specifies neuron type for this layer and takes in
                     type-specific parameters
        """
        
        self.num = num
        self.name = name
        
        if neurontype == None:
            self.neurons = LIFNeuron(self.name, self.num) # set default type as LIFNeuron
        else:
            self.neurons = neurontype 
    

    def plot(self, num=None):       # TODO: add spikes maybe?
        """
        Plots the threshold and voltage versus time of the layer.
        
        parameters
        ----------
        num : specifies how many neurons to plot. randomly plots the values of that many neurons,
              otherwise plots all neurons in the layer
        """
        
        # sample is the number of neurons to show (in case u have a lot and want to see more clearly)
        num_steps = self.Vs.shape[0]
        time = num_steps * self.dt
        
        if num != None:
            indx = np.random.randint(0, self.num + 1, size = (num,))
            plt.figure(figsize=(20,6))
            plt.suptitle(self.name)
            plt.plot(np.arange(time, step=self.dt), self.Vs[:,indx])
            
            plt.xlabel('time [s]')
            plt.ylabel('voltage')
        
            plt.figure(figsize=(20,6))
            plt.plot(np.arange(time, step=self.dt), self.Ths[:,indx])
            plt.xlabel('time [s]')
            plt.ylabel('threshold')        
        
        else:
            plt.figure(figsize=(20,6))
            plt.suptitle(self.name)
            plt.plot(np.arange(time, step=self.dt), self.Vs[:,:])
            
            plt.xlabel('time [s]')
            plt.ylabel('voltage')
        
            plt.figure(figsize=(20,6))
            plt.plot(np.arange(time, step=self.dt), self.Ths[:,:])
            plt.xlabel('time [s]')
            plt.ylabel('threshold')

            
    def plot_wave(self):
        """Plots the fired neurons of each time step of the layer to visualize the spatiotemporal wave"""
        num_steps = self.Sp.shape[0]
        for i in range(1, num_steps):
        # Visualization of Wave
            plt.scatter(self.neurons.nx[:,0], self.neurons.nx[:,1], color = 'k')
            plt.title(self.name + ' t =' + str(i))
            fired = np.argwhere(self.Sp[i,:])
            plt.scatter(self.neurons.nx[fired,0], self.neurons.nx[fired,1], color = 'r')
            plt.show()
            clear_output(wait=True)


###################################################################################################
###################################   NEURON TYPES  ###############################################
###################################################################################################
#
# All neurontype classes need at minimum these four attributes:
#     voltage   : array storing voltage values on the current time step
#     threshold : array storing threshold values on the current time step
#     spiked    : array storing spiked neurons on the current time step
#     nx        : positions (plotting purposes)
# 
# All neurontype classes need at minimum these three methods:
#
#      __init__(self, num, ...): constructor, returns neuron object
#         parameters
#         -----------
#         num : number of neurons
#         ... : any additional parameters specific to that type 
#
#      step_first(self, dt): 
#         Updates the voltage and threshold values of the neuron layer (layer.neurons) using
#         previous values and noise. meant for the first layer in the network
#         parameters (need to be exact)
#         -----------------------------
#         dt : time step 
#
#      step_second(self, x, dt): 
#         Updates the voltage and threshold value of the neuron using previous values and possible
#         inputs from other layers. useful so that we can use the step of any neuron type in the layer
#         parameters (need to be exact)
#         -----------------------------
#         x  : input from the previous layer, calculated as weights * previous layer's spiked vector
#         dt : time step


class LIFNeuron:
    """
    A class to specify a layer as having dynamics of LIFNeurons
    
    attributes
    ----------
    num       : number of neurons in the layer
    ai        : excitation amplitude factor 
    ao        : inhibition amplitude factor
    ri        : excitation radius
    ro        : inhibition radius
    sqR       : square radius
    nx        : positions
    S         : adjacency matrix
    voltage   : voltage values on the current time step
    threshold : threshold values on the current time step
    spiked    : spiked neurons on the current time step
    v_reset   : reset voltage
    thresh    : ReLU activation threshold values
    tau_v      : constant for voltage equation
    tau_th     : constant for threshold equation
    th_plus    : constant for threshold equation
    v_th       : constant for threshold equation
    """

    def __init__(self, num,  ai = 16, ao = 3, ri=2, ro=5 , sqR = 28, thresh_mu = 40, thresh_sig = 2,
                 tau_v = 2, tau_th = 60, th_plus = 9, v_th = 1):
        """
        Returns a LIFNeuron object
        
        parameters
        ----------
        num        : number of neurons in the layer
        ai         : excitation amplitude factor 
        ao         : inhibition amplitude factor
        ri         : excitation radius
        ro         : inhibition radius
        sqR        : square radius
        thresh_mu  : mu value for normally distributed ReLU activation thresh values
        thresh_sig : sigma value for normally distributed ReLU activation thresh values
        tau_v      : constant for voltage equation
        tau_th     : constant for threshold equation
        th_plus    : constant for threshold equation
        v_th       : constant for threshold equation
        """
        # attributes that can take parameters
        self.num = num
        self.ai = ai
        self.ao=ao
        self.ri=ri
        self.ro=ro
        self.sqR = sqR
        self.nx = self.sqR * np.random.rand(num, 2)
        D = squareform(pdist(self.nx))
        self.tau_v = tau_v
        self.tau_th = tau_th
        self.th_plus = th_plus
        self.v_th = v_th
        
        # attributes that don't take in parameters
        self.S = adjacency(D, ai, ao, ri, ro)
        self.voltage = np.zeros((self.num, 1))
        self.threshold = np.ones((self.num, 1))
        self.spiked = np.zeros((self.num, 1))
        self.v_reset = .1 * np.random.randn(self.num, 1)**2
        self.thresh = np.random.normal(thresh_mu, thresh_sig, (self.num,1))
        

    def step_first(self, dt):
        """
        Updates the voltage and threshold values of the neuron layer (layer.neurons) using
        previous values and noise. meant for the first layer in the network
        
        parameters:
        -----------
        dt : time step
        """
          
        eta = 3 * np.random.randn(self.num, 1)
        U = np.matmul(self.S,self.spiked)
        dV = 1/self.tau_v * (-1* self.voltage + U + eta)
        self.voltage[:] += dV * dt

        # step threshold voltage (theta)
        dTh = (1/self.tau_th * (self.v_th - self.threshold) * (1-self.spiked) + self.th_plus * self.spiked) 
        self.threshold[:] += dTh * dt

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = self.voltage > self.threshold
        self.spiked[:] = spiked_mask * (1)

        # set spiked voltages to v_reset and rectify negative voltages to a floor of min_voltage
        self.voltage[self.voltage < 0] = 0 # min_voltage 
        self.voltage[spiked_mask] = self.v_reset[spiked_mask]
        

    
    def step_second(self, x, dt):
        """
        Updates the voltage and threshold value of the neuron using previous values and possible
        inputs from other layers. useful so that we can use the step of any neuron type in the layer
        
        parameters
        -----------
        x  : input from the previous layer, calculated as weights * previous layer's spiked vector
        dt : time step
        """
        # competition rule
        y = np.maximum(x - self.thresh, 0) # ReLU activation function 
        win, maxInd = np.max(y), np.argmax(y)
        y[y < win] = 0 # only allow winner to participate (make all other entries 0)
        self.thresh = self.thresh + dt * (0.005 * y)

        U = np.matmul(self.S, self.spiked) + y # np.matmul(self.S_x, y)
        dV = 1/self.tau_v * (-1* self.voltage + U)
        self.voltage[:] += dV * dt

        # step threshold voltage (theta)
        dTh = (1/self.tau_th * (self.v_th - self.threshold) * (1-self.spiked) + self.th_plus * self.spiked) 
        self.threshold[:] += dTh * dt

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = self.voltage > self.threshold
        self.spiked[:] = spiked_mask * (1)

        # set spiked voltages to v_reset and rectify negative voltages to a floor of min_voltage
        self.voltage[self.voltage < 0] = 0 # min_voltage 
        self.voltage[spiked_mask] = self.v_reset[spiked_mask]
        
        
###################################################################################################
########################    LEARNING RULES  #######################################################
###################################################################################################
#
# All neurontype classes need at minimum these two methods:
#      __init__(self, ...): constructor, returns neurontype object
#
#      update(self, weights, pre, post, dt)
#         parameters (need to be exact)
#         -----------------------------
#         weights : weight matrix
#         pre     : pre layer
#         post    : post layer
#         dt      : time step


class STDP:
    """
    A class for the modified STDP learning rule: 
    learning rate * outer product of pre and post spiking output signals of the two layers that the
    weights connect
    
    attributes
    -----------
    learning_rate : learning rate constant
    """
    
    def __init__(self, learning_rate = .1):
        """Creates an STDP rule object with the specified learning rate"""
        self.learning_rate = learning_rate
    
    def update(self, weights, pre, post, dt):
        """
        Takes in two layers and returns the updated weight matrix.
        
        parameters
        ----------
        weights : weight matrix
        pre     : pre layer
        post    : post layer
        dt      : time step
        """
        delta = dt * np.matmul(self.learning_rate * pre.neurons.spiked, np.transpose(post.neurons.spiked))
        weights += delta
        return weights
        
    
###################################################################################################
##########################      FUNCTIONS     #####################################################
###################################################################################################
#
# Miscellaneous functions used in the module

def adjacency(D, ai, ao, ri, ro):
    """
    Returns intra-layer adjacency matrix encoding the spatial connectivity 
    of neurons within the layer

    parameters
    ----------
    D  : distance matrix
    ai : excitation amplitude factor
    a0 : inhibition amplitude factor
    ri : float
         excitation radius
    ro : float
         inhibition radius
    """
    
    S = ai * (D < ri) - (ao * (D > ro) *  np.exp(-D/10))  
    S = S - np.diag(np.diag(S)) 

    return S
