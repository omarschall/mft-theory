import numpy as np
import time
from utils import *

class Simulation:
    """Simulation class"""

    def __init__(self, rnn):
        """Initialize with the rnn under study"""

        self.rnn = rnn

    def run(self, T, ode_method, x_init, monitors=[], I_ext=None,
            verbose=True, T_monitor=None, T_monitor_interval=1,
            compute_phi_lpf=False):
        """Run the simulation for a specific time interval and external inputs.

        Args:
            T (float): The amount of time in time units (not time steps) for
                the simulation to run
            ode_method (ODE_Method): An instance of ODE_Method for numerically
                solving the ODE. The discretization structure for the rest of
                the module is inherited from the ode_method's parameters.
            x_init (numpy array): A numpy array of shape (self.rnn.N) that
                specifies the initial state of the system (at time step -1).
            monitors (list): A list of strings for addressing simluation (or
                nested) attributes and storing over the simulation).
            I_ext (numpy array or None): External currents for each time step.
                Discretization must much that of ode_method.
            verbose (boolean): Flag indicating whether to print progress reports
                or not.
            T_monitor (float or None): Time after which the simulation starts
                tracking internal data via monitors.
            compute_phi_lpf (boolean): Flag indicating whether to compute the
                low-pass filtered version of phi or not."""

        # Store core attributes
        self.T = T
        self.ode_method = ode_method
        self.x_init = x_init
        self.rnn.x = x_init.copy()
        self.rnn.phi = self.rnn.activation.f(self.rnn.x)
        self.time_vector = np.arange(0, T, ode_method.dt)
        self.total_time_steps = len(self.time_vector)
        self.report_interval = max(self.total_time_steps // 10, 1)
        self.verbose = verbose
        self.T_monitor = T_monitor
        self.T_monitor_interval = T_monitor_interval
        self.compute_phi_lpf = compute_phi_lpf
        self.phi_lpf = 0

        # Initialize monitors
        self.mons = {k: [] for k in monitors}

        # Initialize start time
        self.start_time = time.time()

        for i_t, t in enumerate(self.time_vector):

            self.i_t = i_t
            self.t = t

            # Check for external input
            if I_ext is not None:
                I = I_ext[i_t]
            else:
                I = None

            # Update network state
            self.rnn.x = self.ode_method.next_state(self.rnn.x, self.rnn.x_dot(I=I))
            if self.compute_phi_lpf:
                self.phi_lpf = self.phi_lpf + self.ode_method.dt * (-self.phi_lpf + self.rnn.phi)

            # Update monitors
            if self.T_monitor is None and (i_t % self.T_monitor_interval)==0:
                self.update_monitors()
                self.get_radii_and_norms()
            else:
                if t > self.T_monitor and (i_t % self.T_monitor_interval)==0:
                    self.update_monitors()
                    self.get_radii_and_norms()
                else:
                    pass

            # Make report if conditions are met
            if (self.i_t % self.report_interval == 0 and
                    self.i_t > 0 and
                    self.verbose):
                self.report_progress()

        self.monitors_to_arrays()

    def report_progress(self):
        """"Reports progress at specified interval, including test run
        performance if specified."""

        progress = np.round((self.i_t / self.total_time_steps) * 100, 2)
        time_elapsed = np.round(time.time() - self.start_time, 1)

        summary = '\rProgress: {}% complete \nTime Elapsed: {}s \n'
        print(summary.format(progress, time_elapsed))

    def update_monitors(self):
        """Loops through the monitor keys and appends current value of any
        object's attribute found."""

        for key in self.mons:
            try:
                self.mons[key].append(rgetattr(self, key))
            except AttributeError:
                pass

    def get_radii_and_norms(self):
        """Calculates the spectral radii and/or norms of any monitor keys
        where this is specified."""

        for feature, func in zip(['radius', 'norm'],
                                 [get_spectral_radius, norm]):
            for key in self.mons:
                if feature in key:
                    attr = key.split('-')[0]
                    self.mons[key].append(func(rgetattr(self, attr)))

    def monitors_to_arrays(self):
        """Recasts monitors (lists by default) as numpy arrays for ease of use
        after running."""

        for key in self.mons:
            try:
                self.mons[key] = np.array(self.mons[key])
            except ValueError:
                pass