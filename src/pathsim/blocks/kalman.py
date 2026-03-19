#########################################################################################
##
##                               KALMAN FILTER BLOCK 
##                                (blocks/kalman.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..utils.register import Register
from ..events.schedule import Schedule


# BLOCKS ================================================================================

class KalmanFilter(Block):
    """Discrete-time Kalman filter for state estimation.
    
    Implements the standard Kalman filter algorithm to estimate the state of a 
    linear dynamic system from noisy measurements. The filter recursively updates 
    state estimates by combining predictions from a system model with incoming 
    measurements, weighted by their respective uncertainties.
    
    The filter processes measurements at each time step through a two-stage process:
    prediction (using the system model) and update (incorporating measurements).
    
    The system model is:
    
    .. math::
        x_{k+1} = F x_k + B u_k + w_k
        
        z_k = H x_k + v_k
    
    where :math:`w_k \\sim \\mathcal{N}(0, Q)` is process noise and 
    :math:`v_k \\sim \\mathcal{N}(0, R)` is measurement noise.
    
    At each time step, the filter performs:
    
    **Prediction:**
    
    .. math::
        \\hat{x}_{k|k-1} = F \\hat{x}_{k-1} + B u_k
        
        P_{k|k-1} = F P_{k-1} F^T + Q
    
    **Update:**
    
    .. math::
        y_k = z_k - H \\hat{x}_{k|k-1}
        
        S_k = H P_{k|k-1} H^T + R
        
        K_k = P_{k|k-1} H^T S_k^{-1}
        
        \\hat{x}_k = \\hat{x}_{k|k-1} + K_k y_k
        
        P_k = (I - K_k H) P_{k|k-1}
    
    Note
    ----
    The block expects inputs in the following order:
    
    - First m inputs: measurements :math:`z`
    - Next p inputs (if B is provided): control inputs :math:`u`
    
    The block outputs the n-dimensional state estimate :math:`\\hat{x}`.
   
    Parameters
    ----------
    F : ndarray
        State transition matrix (n x n). Describes how the state evolves from one
        time step to the next.
    H : ndarray
        Measurement matrix (m x n). Maps the state space to the measurement space.
    Q : ndarray
        Process noise covariance matrix (n x n). Represents uncertainty in the
        system model.
    R : ndarray
        Measurement noise covariance matrix (m x m). Represents uncertainty in
        the measurements.
    B : ndarray, optional
        Control input matrix (n x p). Maps control inputs to state changes.
        Default is None (no control input).
    x0 : ndarray, optional
        Initial state estimate (n,). Default is zero vector.
    P0 : ndarray, optional
        Initial error covariance matrix (n x n). Default is identity matrix.
    
    Attributes
    ----------
    x : ndarray
        Current state estimate :math:`\\hat{x}_k`
    P : ndarray
        Current error covariance matrix :math:`P_k`
    n : int
        State dimension
    m : int
        Measurement dimension
    p : int
        Control input dimension 
    """

    def __init__(self, F, H, Q, R, B=None, x0=None, P0=None, dt=None):
        super().__init__()

        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B

        # Sampling
        self.dt = dt 

        # Dimensions
        self.n, _ = F.shape  # state dimension
        self.m, _ = H.shape  # measurement dimension
        _, self.p = (0, 0) if B is None else B.shape # control dimension
            
        # Initial states
        self.x = np.zeros(self.n) if x0 is None else x0
        self.P = np.eye(self.n) if P0 is None else P0

        # Initialize io
        self.inputs = Register(size=self.m+self.p)
        self.outputs = Register(size=self.n)

        # Scheduled event if 'dt' is provided
        if self.dt is not None:
            self.events = [
                Schedule(
                    t_period=self.dt,
                    func_act=lambda _: self._kf_update()
                    )
                ]


    def __len__(self):
        #no passthrough by definition
        return 0


    def to_checkpoint(self, prefix, recordings=False):
        """Serialize Kalman filter state estimate and covariance."""
        json_data, npz_data = super().to_checkpoint(prefix, recordings=recordings)
        npz_data[f"{prefix}/kf_x"] = self.x
        npz_data[f"{prefix}/kf_P"] = self.P
        return json_data, npz_data


    def load_checkpoint(self, prefix, json_data, npz):
        """Restore Kalman filter state estimate and covariance."""
        super().load_checkpoint(prefix, json_data, npz)
        if f"{prefix}/kf_x" in npz:
            self.x = npz[f"{prefix}/kf_x"]
        if f"{prefix}/kf_P" in npz:
            self.P = npz[f"{prefix}/kf_P"]


    def _kf_update(self):
        """Perform one Kalman filter update step."""

        # Unpack inputs
        zu = self.inputs.to_array()
        z, u = np.split(zu, [self.m])

        # Prediction
        x_pred = self.F @ self.x + (self.B @ u if self.B is not None else 0.0)
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Innovation
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R   
        
        # Kalman gain
        K = np.linalg.solve(S.T, (P_pred @ self.H.T).T).T
        
        # Update state
        self.x = x_pred + K @ y        
        self.P = (np.eye(self.n) - K @ self.H) @ P_pred 

        # Update outputs
        self.outputs.update_from_array(self.x)


    def sample(self, t, dt):
        """Sample after successful timestep.

        Updates the internal state estimate using the current measurements and
        control inputs, then outputs the updated state estimate.

        Parameters
        ----------
        t : float
            evaluation time for sampling
        dt : float
            integration timestep
        """
        if self.dt is None:
            self._kf_update()