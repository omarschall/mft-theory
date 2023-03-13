class ODE_Method:
    """Parent class for all ODE methods"""

    def __init__(self, dt):
        """Initialize any ODE method with its core time
        discretization"""

        self.dt = dt

    def next_state(self, state, state_derviative):
        """Each method fills in its own method for getting
        the next discrete state in terms of the current state"""

        pass