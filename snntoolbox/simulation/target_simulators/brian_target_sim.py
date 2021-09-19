# -*- coding: utf-8 -*-
"""Building and simulating spiking neural networks using Brian.

All the work is done in pyNN, so we simply redirect :py:mod:`brian_target_sim`
to :py:mod:`pyNN_target_sim` here.

@author: rbodo
"""

# noinspection PyUnresolvedReferences
from snntoolbox.simulation.target_simulators.pyNN_target_sim import SNN
