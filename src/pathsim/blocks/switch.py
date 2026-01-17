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
    """Switch block that selects between its inputs and copies 
    one of them to the output. 

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
    
    Sets block output depending on `self.state` like this:

    .. code-block::

        state == None -> outputs[0] = 0

        state == 0 -> outputs[0] = inputs[0]

        state == 1 -> outputs[0] = inputs[1]

        state == 2 -> outputs[0] = inputs[2]
    
        ...

    Parameters
    ----------
    state : int, None
        state of the switch
    
    """

    input_port_labels = None
    output_port_labels = {"out":0}

    def __init__(self, state=None):
        super().__init__()

        self.state = state


    def __len__(self):
        """Algebraic passthrough only possible if state is defined"""
        return 0 if (self.state is None or not self._active) else 1


    def select(self, state=0):
        """
        This method is unique to the `Switch` block and intended 
        to be used from outside the simulation level for selecting 
        the input ports for the switch state.
    
        This can be achieved for example with the event management 
        system and its callback/action functions.

        Parameters
        ---------
        state : int, None
            switch state / input port selection
        """
        self.state = state


    def update(self, t):
        """Update switch output depending on inputs and switch state.

        Parameters
        ----------
        t : float
            evaluation time
        """
        
        #early exit without error control
        if self.state is None: self.outputs[0] = 0.0
        else: self.outputs[0] = self.inputs[self.state]
