# Custom Learning Rule
# Subclass of LearningRuleType

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import pdist, squareform
from IPython.display import clear_output

import nengo
from nengo.rc import rc
from nengo.params import Parameter, NumberParam, NdarrayParam, Default
from nengo.neurons import settled_firingrate, LIFRate 
from nengo.builder.operator import Operator, Copy
from nengo.builder import Builder
from nengo.builder.signal import Signal
from nengo.synapses import Lowpass, SynapseParam
from nengo.builder.learning_rules import get_pre_ens, get_post_ens, build_or_passthrough


class STDP(nengo.learning_rules.LearningRuleType):
    """STDP Hebbian learning rule.
    Modifies connection weights according to the Hebbian STDP rule.
    Notes
    -----
    The rule is dependent on pre and post neural activities,
    not decoded values, and so is not affected by changes in the
    size of pre and post ensembles. However, if you are decoding from
    the post ensemble, the Oja rule will have an increased effect on
    larger post ensembles because more connection weights are changing.
    In these cases, it may be advantageous to scale the learning rate
    on the rule by ``1 / post.n_neurons``.
    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which weights will be adjusted.
    pre_synapse : `.Synapse`, optional
        Synapse model used to filter the pre-synaptic activities.
    post_synapse : `.Synapse`, optional
        Synapse model used to filter the post-synaptic activities.
        If None, ``post_synapse`` will be the same as ``pre_synapse``.
    Attributes
    ----------
    learning_rate : float
    A scalar indicating the rate at which weights will be adjusted.
    post_synapse : `.Synapse`
        Synapse model used to filter the post-synaptic activities.
    pre_synapse : `.Synapse`
        Synapse model used to filter the pre-synaptic activities.
    """

    modifies = "weights"
    probeable = ("pre_filtered", "post_filtered", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)

    def __init__(
        self,
        learning_rate=Default,
        pre_synapse=Default,
        post_synapse=Default,
    ):
        super().__init__(learning_rate, size_in=0)

        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )

    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)
    

    
# Simulator Operator that implements learning rule

class SimSTDP(Operator):
    r"""Calculate connection weight change according to the STDP Hebbian rule.
    Implements the Hebbian learning rule of the form
    .. math:: \Delta \omega_{ij} = \kappa (a_i a_j)
    where
    * :math:`\kappa` is a scalar learning rate,
    * :math:`a_i` is the activity of a presynaptic neuron,
    * :math:`a_j` is the activity of a postsynaptic neuron,
    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str, optional
        A label associated with the operator, for debugging purposes.
    Attributes
    ----------
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, post_filtered, weights]``
    4. updates ``[delta]``
    """

    def __init__(
        self, pre_filtered, post_filtered, delta, learning_rate, tag=None
    ):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def _descstr(self):
        return "pre=%s, post=%s -> %s" % (
            self.pre_filtered,
            self.post_filtered,
            self.delta,
        )

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        def step_simSTDP():
            # perform update
            delta[...] += alpha* np.outer(post_filtered, pre_filtered)

        return step_simSTDP
    
    
# Build function for new learning type

@Builder.register(STDP)
def build_STDP(model, STDP, rule):
    """Builds a `.STDP` object into a model.
    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimSTDP` operator to the model to calculate the delta.
    Parameters
    ----------
    model : Model
        The model to build into.
    STDP : STDP
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Oja` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, STDP.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, STDP.post_synapse, post_activities)

    model.add_op(
        SimSTDP(
            pre_filtered,
            post_filtered,
            model.sig[rule]["delta"],
            learning_rate=STDP.learning_rate,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
    

    
    
    
# Nora's learning rule version, modeled after Oja
####################################################
####################################################

class STDP2(nengo.learning_rules.LearningRuleType):
    """STDP2 Hebbian learning rule.
    Modifies connection weights according to the Hebbian STDP2 rule.
    Notes
    -----
    The rule is dependent on pre and post neural activities,
    not decoded values, and so is not affected by changes in the
    size of pre and post ensembles. However, if you are decoding from
    the post ensemble, the Oja rule will have an increased effect on
    larger post ensembles because more connection weights are changing.
    In these cases, it may be advantageous to scale the learning rate
    on the rule by ``1 / post.n_neurons``.
    Parameters
    ----------
    learning_rate : float, optional
        A scalar indicating the rate at which weights will be adjusted.
    pre_synapse : `.Synapse`, optional
        Synapse model used to filter the pre-synaptic activities.
    post_synapse : `.Synapse`, optional
        Synapse model used to filter the post-synaptic activities.
        If None, ``post_synapse`` will be the same as ``pre_synapse``.
    Attributes
    ----------
    learning_rate : float
    A scalar indicating the rate at which weights will be adjusted.
    post_synapse : `.Synapse`
        Synapse model used to filter the post-synaptic activities.
    pre_synapse : `.Synapse`
        Synapse model used to filter the pre-synaptic activities.
    """

    modifies = "weights"
    probeable = ("pre_filtered", "post_filtered", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1)
    pre_synapse = SynapseParam("pre_synapse", default=None, readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)

    def __init__(
        self,
        learning_rate=Default,
        pre_synapse=Default,
        post_synapse=Default,
    ):
        super().__init__(learning_rate, size_in=0)

        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )

    @property
    def _argreprs(self):
        return _remove_default_post_synapse(super()._argreprs, self.pre_synapse)
    

    
# Simulator Operator that implements learning rule

class SimSTDP2(Operator):
    r"""Calculate connection weight change according to the STDP2 Hebbian rule.
    Implements the Hebbian learning rule of the form
    .. math:: \Delta \omega_{ij} = \kappa (a_i a_j)
    where
    * :math:`\kappa` is a scalar learning rate,
    * :math:`a_i` is the activity of a presynaptic neuron,
    * :math:`a_j` is the activity of a postsynaptic neuron,
    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str, optional
        A label associated with the operator, for debugging purposes.
    Attributes
    ----------
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, post_filtered, weights]``
    4. updates ``[delta]``
    """

    def __init__(
        self, pre_filtered, post_filtered, delta, learning_rate, tag=None
    ):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def _descstr(self):
        return "pre=%s, post=%s -> %s" % (
            self.pre_filtered,
            self.post_filtered,
            self.delta,
        )

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        def step_simSTDP2():
            # perform update
#             delta[...] += alpha* np.outer(post_filtered, .8 * pre_filtered)
            delta[...] += .1
        return step_simSTDP2
    
    
# Build function for new learning type

@Builder.register(STDP2)
def build_STDP2(model, STDP2, rule):
    """Builds a `.STDP2` object into a model.
    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimSTDP2` operator to the model to calculate the delta.
    Parameters
    ----------
    model : Model
        The model to build into.
    STDP2 : STDP2
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Oja` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, STDP2.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, STDP2.post_synapse, post_activities)

    model.add_op(
        SimSTDP2(
            pre_filtered,
            post_filtered,
            model.sig[rule]["delta"],
            learning_rate=STDP2.learning_rate,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
   