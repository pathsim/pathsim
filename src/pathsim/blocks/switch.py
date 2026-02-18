#########################################################################################
##
##                                   SWITCH BLOCK
##                                (blocks/switch.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from ._block import Block


# BLOCK DEFINITION ======================================================================

class Switch(Block):
    """Switch block that selects between its inputs.

    Example
    -------
    The block is initialized like this:

    .. code-block:: python 
        
        #default None -> no passthrough 
        s1 = Switch()

        #selecting port 2 as passthrough
        s2 = Switch(2)
    
        #change the state of the switch to port 3
        s2.select(3)
    
    Sets block output depending on `self.switch_state` like this:

    .. code-block::

        switch_state == None -> outputs[0] = 0

        switch_state == 0 -> outputs[0] = inputs[0]

        switch_state == 1 -> outputs[0] = inputs[1]

        switch_state == 2 -> outputs[0] = inputs[2]
    
        ...

    Parameters
    ----------
    switch_state : int, None
        state of the switch
    
    """

    input_port_labels = None
    output_port_labels = {"out":0}

    def __init__(self, switch_state=None):
        super().__init__()

        self.switch_state = switch_state


    def __len__(self):
        """Algebraic passthrough only possible if switch_state is defined"""
        return 0 if (self.switch_state is None or not self._active) else 1


    def select(self, switch_state=0):
        """
        This method is unique to the `Switch` block and intended 
        to be used from outside the simulation level for selecting 
        the input ports for the switch state.
    
        This can be achieved for example with the event management 
        system and its callback/action functions.

        Parameters
        ---------
        switch_state : int, None
            switch state / input port selection
        """
        self.switch_state = switch_state


    def update(self, t):
        """Update switch output depending on inputs and switch state.

        Parameters
        ----------
        t : float
            evaluation time
        """
        
        #early exit without error control
        if self.switch_state is None: self.outputs[0] = 0.0
        else: self.outputs[0] = self.inputs[self.switch_state]
