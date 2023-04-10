from ode_methods.ODE_Method import ODE_Method

class RK4(ODE_Method):
    """Simplest ODE method, forward standard Euler"""

    def __init__(self, dt, rnn):
        """Initialize with the core time discretization and also the rnn
        being integrated"""

        super().__init__(dt)
        self.rnn = rnn

    def next_state(self, state, state_derivative):
        """Update next state as sum of previous plus derivative
        scaled by dt"""

        k_1 = self.dt * state_derivative
        k_2 = self.dt * self.rnn.x_dot(x=state + 1/2 * k_1)
        k_3 = self.dt * self.rnn.x_dot(x=state + 1/2 * k_2)
        k_4 = self.dt * self.rnn.x_dot(x=state + k_3)

        return state + 1/6 * k_1 + 1/3 * (k_2 + k_3) + 1/6 * k_4