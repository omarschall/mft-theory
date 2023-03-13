from ode_methods.ODE_Method import ODE_Method

class Euler(ODE_Method):
    """Simplest ODE method, forward standard Euler"""

    def __init__(self, dt):
        """Initialize with just the core time discretization"""

        super().__init__(dt)

    def next_state(self, state, state_derivative):
        """Update next state as sum of previous plus derivative
        scaled by dt"""

        return state + self.dt * state_derivative