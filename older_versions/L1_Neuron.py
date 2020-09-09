import numpy as np
from nengo.neurons import LIFRate, NeuronType
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import pdist, squareform
from IPython.display import clear_output

import nengo
from nengo.params import Parameter, NumberParam, NdarrayParam
from nengo.neurons import settled_firingrate   

class CustomLIF(nengo.neurons.NeuronType):
    """Spiking version of the leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    min_voltage : float
        Minimum value for the membrane voltage. If ``-np.inf``, the voltage
        is never clipped.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    num : int
        Number of neurons in the layer.
    S : ndarray
        Intra-layer adjacency matrix.
    """

    probeable = ("spikes", "voltage", "refractory_time", "threshold")

    min_voltage = NumberParam("min_voltage", high=0)
    tau_rc = NumberParam("tau_rc", low=0, low_open=True)
    tau_ref = NumberParam("tau_ref", low=0)
    amplitude = NumberParam("amplitude", low=0, low_open=True)
    num = NumberParam("num")
    Ss = NdarrayParam("Ss")  # adjacency matrix
    piT_th = NumberParam("piT_th")
    piTV_plus = NumberParam("piTV_plus")
    piV_th = NumberParam("piV_th")
    piV_reset = NdarrayParam("piV_reset")  # noise on activity field
    nxs = NdarrayParam("nxs")
    Vt = NumberParam("Vt")
    
    
    
    def __init__(
        self, 
        Ss, 
        num, 
        tau_rc, 
        piT_th,
        piTV_plus,
        piV_th,
        piV_reset,
        nxs,
        Vt,
        tau_ref=0.002, 
        min_voltage=0, 
        amplitude=1,
        counter = 0,
        wave = True
        
        #current_collect
    ):
        super().__init__()
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.amplitude = amplitude
        self.min_voltage = min_voltage
        self.num = num
        self.Ss = Ss
        self.nxs = nxs
        self.piT_th=piT_th
        self.piTV_plus=piTV_plus
        self.piV_th=piV_th
        self.piV_reset=piV_reset
        self.Vt= Vt
        self.counter = counter
        self.wave = wave

    def gain_bias(self, max_rates, intercepts):
        """Analytically determine gain, bias."""
        gain = np.ones((self.num,))
        bias = np.zeros((self.num,))
        return gain, bias
    

    def max_rates_intercepts(self, gain, bias):
        """Compute the inverse of gain_bias."""
        intercepts = (1 - bias) / gain
        max_rates = 1.0 / (
            self.tau_ref - self.tau_rc * np.log1p(1.0 / (gain * (intercepts - 1) - 1))
        )
        if not np.all(np.isfinite(max_rates)):
            warnings.warn(
                "Non-finite values detected in `max_rates`; this "
                "probably means that `gain` was too small."
            )
        return max_rates, intercepts


    def rates(self, x, gain, bias):
        """Always use LIFRate to determine rates."""
        J = self.current(x, gain, bias)
        out = np.zeros_like(J)
        # Use LIFRate's step_math explicitly to ensure rate approximation
        LIFRate.step_math(self, dt=1, J=J, output=out)
        return out


    def step_math(self, dt, J, spiked, voltage, refractory_time, threshold):        
        # reduce all refractory times by dt
        refractory_time -= dt
#         print("we are on time step" + str(self.counter))
#         print("BEFORE UPDATE")
#         print("voltage:")
#         print(voltage)
#         print("current:")
#         print(J)
#         voltage[:] = J

        
        
        # step voltage
#         print("RETINA")
#         print("this is np.matmul(spiked, self.Ss):{}".format(np.matmul(spiked, self.Ss)))
        
        U = np.matmul(spiked, self.Ss)
        eta = 3*np.random.randn(self.num,) / self.Vt
        dV = -1*voltage + U + eta
        voltage[:] += dV * dt
        

#         print('final voltage = {}'.format(voltage))
                
        # step threshold voltage (theta)
        dTh = self.piT_th*(self.piV_th-threshold)*(1-spiked)+self.piTV_plus*spiked
        threshold[:] += dTh * dt
        print('threshold = {}'.format(threshold))
        print('threshold shape = {}'.format(threshold.shape))

        print('spiked shape= {}'.format(spiked.shape))
        
        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > threshold
        spiked[:] = spiked_mask * (self.amplitude)
#         print('RETINA spiked mask= {}'.format(spiked_mask))
        
        if self.wave == True:
            
            # Visualization of Wave
            plt.scatter(self.nxs[:,0],self.nxs[:,1], color = 'k')
            plt.title('Ret-Wave t =' + str(self.counter))
            fired = np.argwhere(spiked)
            plt.scatter(self.nxs[fired,0], self.nxs[fired,1], color = 'r')
            
            plt.show()
            clear_output(wait=True)
        self.counter += 1


        # set spiked voltages to v_reset, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < self.min_voltage] = self.min_voltage 
        voltage[spiked_mask] = self.piV_reset[spiked_mask]
        refractory_time[spiked_mask] = self.tau_ref
#         print("AFTER UPDATE")
#         print("voltage:")
#         print(voltage)
#         print("____________________________________")
        
from nengo.builder.operator import Operator

class SimCustomLIF(Operator):
    """Set a neuron model output for the given input current.

    Implements ``neurons.step_math(dt, J, output, *states)``.

    Parameters
    ----------
    neurons : NeuronType
        The `.NeuronType`, which defines a ``step_math`` function.
    J : Signal
        The input current.
    output : Signal
        The neuron output signal that will be set.
    states : list, optional
        A list of additional neuron state signals set by ``step_math``.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    J : Signal
        The input current.
    neurons : NeuronType
        The `.NeuronType`, which defines a ``step_math`` function.
    output : Signal
        The neuron output signal that will be set.
    states : list
        A list of additional neuron state signals set by ``step_math``.
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[output] + states``
    2. incs ``[]``
    3. reads ``[J]``
    4. updates ``[]``
    """
    
    def __init__(self, neurons, J, output, states=None, tag=None):
        super().__init__(tag=tag)
        self.neurons = neurons

        self.sets = [output] + ([] if states is None else states)
        self.incs = []
        self.reads = [J]
        self.updates = []

    @property
    def J(self):
        return self.reads[0]

    @property
    def output(self):
        return self.sets[0]

    @property
    def states(self):
        return self.sets[1:]

    def _descstr(self):
        return "%s, %s, %s" % (self.neurons, self.J, self.output)

    def make_step(self, signals, dt, rng):
        J = signals[self.J]
        output = signals[self.output]
        states = [signals[state] for state in self.states]

        def step_simcustomlif():
            self.neurons.step_math(dt, J, output, *states)

        return step_simcustomlif
from nengo.builder import Builder
from nengo.builder.operator import Copy
from nengo.builder.signal import Signal
from nengo.rc import rc


@Builder.register(CustomLIF)
def build_customlif(model, neuron_type, neurons):
    """Builds a `.LIF` object into a model.

    In addition to adding a `.SimNeurons` operator, this build function sets up
    signals to track the voltage and refractory times for each neuron.

    Parameters
    ----------
    model : Model
        The model to build into.
    neuron_type : CustomLIF
        Neuron type to build.
    neuron : Neurons
        The neuron population object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.LIF` instance.
    """

    model.sig[neurons]["voltage"] = Signal(
        shape=neurons.size_in, name="%s.voltage" % neurons
    )
    model.sig[neurons]["refractory_time"] = Signal(
        shape=neurons.size_in, name="%s.refractory_time" % neurons
    )
    model.sig[neurons]["threshold"] = Signal(
        shape=neurons.size_in, name= "%s.threshold" % neurons
    )
    model.add_op(
        SimCustomLIF(
            neurons=neuron_type,
            J=model.sig[neurons]["in"],
            output=model.sig[neurons]["out"],
            states=[
                model.sig[neurons]["voltage"],
                model.sig[neurons]["refractory_time"],
                model.sig[neurons]["threshold"],
            ],
        )
    )
