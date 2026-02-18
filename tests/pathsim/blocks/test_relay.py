########################################################################################
##
##                                  TESTS FOR
##                             'blocks.relay.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.relay import Relay
from pathsim.events.zerocrossing import ZeroCrossingUp, ZeroCrossingDown


# TESTS ================================================================================

class TestRelay(unittest.TestCase):
    """
    Test the implementation of the 'Relay' block class
    """

    def test_init_default(self):

        R = Relay()

        self.assertEqual(R.threshold_up, 1.0)
        self.assertEqual(R.threshold_down, 0.0)
        self.assertEqual(R.value_up, 1.0)
        self.assertEqual(R.value_down, 0.0)

        #check events are created
        self.assertEqual(len(R.events), 2)
        self.assertIsInstance(R.events[0], ZeroCrossingUp)
        self.assertIsInstance(R.events[1], ZeroCrossingDown)


    def test_init_custom(self):

        R = Relay(
            threshold_up=21.0,
            threshold_down=19.0,
            value_up=0.0,
            value_down=1.0
        )

        self.assertEqual(R.threshold_up, 21.0)
        self.assertEqual(R.threshold_down, 19.0)
        self.assertEqual(R.value_up, 0.0)
        self.assertEqual(R.value_down, 1.0)


    def test_len(self):

        R = Relay()
        self.assertEqual(len(R), 0)


    def test_update(self):

        #update is a no-op for relay (events handle switching)
        R = Relay()
        R.inputs[0] = 5.0
        R.update(0)

        #output unchanged since update is a no-op
        self.assertEqual(R.outputs[0], 0.0)


    def test_event_functions(self):

        R = Relay(threshold_up=2.0, threshold_down=-1.0)

        #check up-crossing event function: input - threshold_up
        R.inputs[0] = 5.0
        evt_up = R.events[0].func_evt(0)
        self.assertEqual(evt_up, 3.0)  # 5 - 2

        R.inputs[0] = 0.0
        evt_up = R.events[0].func_evt(0)
        self.assertEqual(evt_up, -2.0)  # 0 - 2

        #check down-crossing event function: input - threshold_down
        R.inputs[0] = 0.0
        evt_down = R.events[1].func_evt(0)
        self.assertEqual(evt_down, 1.0)  # 0 - (-1)

        R.inputs[0] = -5.0
        evt_down = R.events[1].func_evt(0)
        self.assertEqual(evt_down, -4.0)  # -5 - (-1)


    def test_event_actions(self):

        R = Relay(
            threshold_up=1.0,
            threshold_down=-1.0,
            value_up=10.0,
            value_down=-10.0
        )

        #trigger up action
        R.events[0].func_act(0)
        self.assertEqual(R.outputs[0], 10.0)

        #trigger down action
        R.events[1].func_act(0)
        self.assertEqual(R.outputs[0], -10.0)


    def test_thermostat_scenario(self):

        #thermostat: heater on below 19, off above 21
        R = Relay(
            threshold_up=21.0,
            threshold_down=19.0,
            value_up=0.0,
            value_down=1.0
        )

        #temperature rises above 21 -> heater off
        R.events[0].func_act(0)
        self.assertEqual(R.outputs[0], 0.0)

        #temperature drops below 19 -> heater on
        R.events[1].func_act(0)
        self.assertEqual(R.outputs[0], 1.0)


    def test_port_labels(self):

        R = Relay()
        self.assertEqual(R.input_port_labels, {"in": 0})
        self.assertEqual(R.output_port_labels, {"out": 0})


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
