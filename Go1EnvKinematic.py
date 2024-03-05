import time
import copy
import torch
import warnings
import numpy as np
import casadi as ca
import mujoco as mj
import scipy as scipy
import xml.etree.ElementTree as ET
from torch import nn
from numba import jit, njit
from scipy import optimize
from torch.optim import Adam
from numba import NumbaWarning
from mujoco import viewer as mjv
from numpy import arctan2 as atan2
from numpy import arccos as acos, arcsin as asin
from memory_profiler import profile as mem_profile
from pyquaternion import Quaternion
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation as R

# This line will suppress all Numba warnings
warnings.filterwarnings("ignore", category=NumbaWarning)
np.random.seed(1)

def quat_rot_new(vec, q, center_of_rotation, inv=False):
    # Convert the quaternion to a scipy Rotation object
    rotation = R.from_quat(q)
    
    # If the inverse flag is set, use the transpose
    if inv:
        # rotation = rotation.inv()
        rotation = rotation.T
    
    # Subtract the center of rotation from the vector, rotate the vector, then add the center of rotation back
    vec_rot = rotation.apply(vec - center_of_rotation) + center_of_rotation
    
    return vec_rot

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def choose_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def timer(f):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = f(*args, **kwargs)
        name = f.__name__
        stop_time = time.time()
        dt = stop_time - start_time
        print("Time took for", name, ":", dt, "seconds.")
        return res
    
    return wrapper

# @jit(nopython=True) 
def quaternion_mult(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return w, x, y, z

# @jit
def quat_rot(vec, q, center_of_rotation=[0, 0, 0], inv=False) -> DeprecationWarning:
    def quaternion_conjugate(quaternion):
        w, x, y, z = quaternion
        return (w, -x, -y, -z)

    def rotate_vector(vector, quaternion):
        # v = (0.0,) + tuple(vector)
        v = np.insert(vector, 0, 0.0)
        q_conjugate = quaternion_conjugate(quaternion)
        if not inv:
            return quaternion_mult(quaternion_mult(q_conjugate, v), quaternion)[1:]
        else: return quaternion_mult(quaternion_mult(quaternion, v), q_conjugate)[1:]

    return np.asarray(rotate_vector(vec-center_of_rotation, q)) + center_of_rotation

def quat_rot_mj(vec, q, center_of_rotation=[0, 0, 0], inv=False):
    to_return = np.zeros(3, )
    if not inv:
        conj_quat = np.zeros(4, )
        mj.mju_negQuat(conj_quat, q)
        mj.mju_rotVecQuat(to_return, vec - center_of_rotation, conj_quat)
        return to_return + center_of_rotation
    else:
        mj.mju_rotVecQuat(to_return, vec - center_of_rotation, q)
        return to_return + center_of_rotation

# @jit(nopython=True) 
def quat2rot(q: np.ndarray) -> DeprecationWarning:
    # Ensure the input is a valid quaternion
    assert np.isclose(np.linalg.norm(q), 1), "Input must be a valid quaternion with unit norm"

    w, x, y, z = q

    # Calculate the elements of the rotation matrix
    R = np.array([[1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]])

    return R.transpose()

def quat2rot_mj(q: np.ndarray) -> np.ndarray:
    to_return = np.zeros(9, )
    mj.mju_quat2Mat(to_return, q)
    return to_return.reshape(3, 3).T

# @jit(nopython=True) 
def rot2eul(R: np.ndarray) -> np.ndarray:
    # Ensure the input is a valid rotation matrix
    assert np.allclose(np.dot(R, R.T), np.eye(3)), "Input must be a valid rotation matrix"

    # Calculate the yaw (z-axis rotation)
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # Calculate the pitch (y-axis rotation)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))

    # Calculate the roll (x-axis rotation)
    roll = np.arctan2(R[2, 1], R[2, 2])

    return np.array([roll, pitch, yaw])

# @jit
def quat2eul(q: np.ndarray) -> np.ndarray:
    # Ensure the input is a valid quaternion
    assert np.isclose(np.linalg.norm(q), 1), "Input must be a valid quaternion with unit norm"

    w, x, y, z = q

    # Calculate the roll (x-axis rotation)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))

    # Calculate the pitch (y-axis rotation)
    pitch_sin = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(pitch_sin, -1, 1))

    # Calculate the yaw (z-axis rotation)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))

    return np.asarray([roll, pitch, yaw])

# @jit
def eul2quat(eul: np.ndarray) -> np.ndarray:
    eul = np.asarray(eul)
    was_1d = False
    if eul.ndim == 1:
        eul = eul[None, :]  # add an extra dimension
        was_1d = True

    # yaw, pitch, roll = eul[2], eul[1] , eul[0]
    roll, pitch, yaw = eul[:,0], eul[:,1], eul[:,2]
    cos_yaw_2 = np.cos(yaw/2)
    sin_yaw_2 = np.sin(yaw/2)
    cos_pitch_2 = np.cos(pitch/2)
    sin_pitch_2 = np.sin(pitch/2)
    cos_roll_2 = np.cos(roll/2)
    sin_roll_2 = np.sin(roll/2)
    
    # Compute quaternion
    w = cos_yaw_2 * cos_pitch_2 * cos_roll_2 + sin_yaw_2 * sin_pitch_2 * sin_roll_2
    x = cos_yaw_2 * cos_pitch_2 * sin_roll_2 - sin_yaw_2 * sin_pitch_2 * cos_roll_2
    y = cos_yaw_2 * sin_pitch_2 * cos_roll_2 + sin_yaw_2 * cos_pitch_2 * sin_roll_2
    z = sin_yaw_2 * cos_pitch_2 * cos_roll_2 - cos_yaw_2 * sin_pitch_2 * sin_roll_2
    
    # return np.array([w, x, y, z])
    quat = np.array([w, x, y, z]).T
    if was_1d:
        quat = quat[0]  # remove the extra dimension
    return quat

# @jit(nopython=True) 
def parabola_trajectory(leg_pos_0, x1, y1, z_max, n, env) -> np.ndarray:
    n = n * 2
    x0, y0, z0 = leg_pos_0[0], leg_pos_0[1], leg_pos_0[2]
    quat = env.robot_data.qpos[3:7]
    com3 = env.robot_data.qpos[:3]
    
    x0_rot, y0_rot, z0_rot = quat_rot_mj(vec=leg_pos_0, q=quat, center_of_rotation=com3, inv=False)
    x1_rot, y1_rot, z1_rot = quat_rot_mj(vec=np.asarray([x1, y1, z0]), q=quat, center_of_rotation=com3, inv=False)
    
    # Calculate the vertex (h, k) of the parabola
    h = (x0 + x1) / 2
    h_rot = (x0_rot + x1_rot) / 2

    # Calculate the corresponding y values using the parabolic equation
    y_values_rot = np.linspace(y0_rot, y1_rot, n)
    y_values = np.linspace(y0, y1, n)
    # y_values = np.linspace(y0, y1, n)
    middle_y = y_values[(len(y_values)//2)]

    h1_rot, y_mid_rot, z_max_rot = quat_rot_mj(vec=np.asarray([h, middle_y, z_max]), q=quat, center_of_rotation=com3, inv=False)

    k = z_max
    k_rot = z_max_rot

    # Calculate the coefficient a of the parabolic equation
    a_rot = (z0_rot - k_rot) / ((x0_rot - h_rot) ** 2)
    
    # Generate n evenly spaced x values between x0 and x1
    x_end = x1_rot + 0.195
    x_values = np.linspace(x0_rot, x_end, n)

    z_values = a_rot * (x_values - h1_rot) ** 2 + k_rot
    if np.diff(z_values)[0] < 0:
        # print('Wrong step trajectory values. Check if needed.')
        traj = np.asarray([[x0, y0, z0]])
        return traj
    
    zero_vec = com3
    traj = np.array([x_values, y_values_rot, z_values]).T
    traj_new = np.asarray([quat_rot_mj(point, quat, center_of_rotation=zero_vec, inv=True) for point in traj])

    return traj_new

# @jit(nopython=True) 
def line(x0, y0, x1, y1, m, n, z, z_end) -> np.ndarray:
    # Calculate y1 using the slope m
    # y1 = m * (x1 - x0) + y0
    
    # Generate n equally spaced x and y coordinates between x0, y0 and x1, y1
    x = np.linspace(x0, x1, n)
    y = np.linspace(y0, y1, n)

    z = np.linspace(z, z_end, n)

    # Combine x and y coordinates into an array of points
    points = np.column_stack((x, y, z))

    return points.transpose()

# @jit(nopython=True) 
def step_line(leg_pos_0, n, dz, quat) -> np.ndarray:
    
    body_0 = quat_rot_mj(vec=leg_pos_0, q=quat)
    z_end = body_0[2] + dz
    z = np.linspace(body_0[2], z_end, n)
    
    points = np.asarray([quat_rot_mj(vec=np.asarray([body_0[0], body_0[1], z_val]), q=quat, inv=True) for z_val in z])

    # Combine x and y coordinates into an array of points
    # points = np.column_stack((x, y, z))

    return points

def generate_3d_circle(p, a, r, num_points=100):
    """
    Generate a circle in 3D space.
    
    Parameters:
    p (array-like): A point on the circle [x, y, z].
    a (array-like): Axis of rotation (unit vector) [a_x, a_y, a_z].
    r (float): Radius of the circle.
    num_points (int): Number of points to generate on the circle.
    
    Returns:
    np.ndarray: An array of shape (num_points, 3), containing coordinates of points on the circle.
    """
    
    # Step 1: Take any vector b not parallel to a
    b = np.array([1.0, 0.0, 0.0]) if np.all(a != [1.0, 0.0, 0.0]) else np.array([0.0, 1.0, 0.0])
    
    # Step 2: Compute orthogonal unit vector u
    u = b - np.dot(a, b) * a
    u = u / np.linalg.norm(u)
    
    # Step 3: Compute orthogonal unit vector v
    v = np.cross(a, u)
    
    # Generate points on the circle
    t_values = np.linspace(0, 2*np.pi, num_points)
    circle_points = np.array([p + r * (np.cos(t) * u + np.sin(t) * v) for t in t_values])
    
    return circle_points

class LQRController:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
    
    def update_matrices(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def calc_P(self):
        # Solve the Riccati equation.
        try:
            return scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        except:
            return np.zeros((self.Q.shape))

    def lqr_gain(self):
        self.P = self.calc_P()
        # Compute the Feedback Gain Mtrix K.
        return np.linalg.inv(self.R + self.B.T @ self.P @ self.B) @ self.B.T @ self.P @ self.A

class PIDController:
    def __init__(self, env, P_gain=None, I_gain=None, D_gain=None):
        # ### Working values:
        self.default_gains = {'P': np.array([100, 100, 200, 100, 100, 200, 100, 100, 200, 100, 100, 200]),
                                'I': np.array([.01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01, .01]),
                                'D': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])*10
                                }

        V_gain = np.array([100, 100, 200, 100, 100, 200, 100, 100, 200, 100, 100, 200])

        self.P_gain = P_gain
        self.I_gain = I_gain
        self.D_gain = D_gain  
        self.V_gain = V_gain
        self.env = env
        self.i_term = 0
        self.dt = self.env.dt
        self.last_pos_err = 0
        self.last_vel_err = 0
    
    def calc_signal(self, des_x, des_v=None):
        p_err = des_x - self.env.robot_data.qpos[7:]
        dp_err = p_err - self.last_pos_err

        if des_v is not None:
            v_err = des_v - self.env.robot_data.qvel[6:]
        else:
            v_err = 0

        p_term = self.P_gain * (p_err)
        dp_term = self.D_gain * v_err
        # self.i_term += p_err * self.I_gain * self.dt
        # torque = p_term + dp_term + self.i_term
        torque = p_term + dp_term

        self.last_pos_err = p_err

        return torque

    def reset_i_term(self, ):
        self.last_pos_err = 0
        self.i_term = 0

class PurePursuit:
    def __init__(self, path, lookahead_distance, wheelbase_length, max_steer_angle):
        self.path = path
        self.path_len = len(path)
        self.lookahead_distance = lookahead_distance
        self.wheelbase_length = wheelbase_length
        self.max_steer_angle = max_steer_angle

        self.last_closest_idx = 0

    def find_closest_point(self, vehicle_position):
            closest_distance = float('inf')
            closest_idx = self.last_closest_idx
            # Forward search
            for i in range(self.last_closest_idx, len(self.path)):
                distance = np.linalg.norm(np.array(vehicle_position) - np.array(self.path[i]))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_idx = i

            # Backward search
            for i in range(self.last_closest_idx, -1, -1):
                distance = np.linalg.norm(np.array(vehicle_position) - np.array(self.path[i]))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_idx = i

            self.last_closest_idx = closest_idx
            return closest_idx

    def find_lookahead_point(self, vehicle_position):
        closest_idx = self.find_closest_point(vehicle_position)
        for i in range(closest_idx, len(self.path)):
            distance = np.linalg.norm(np.array(vehicle_position) - np.array(self.path[i]))
            if distance >= self.lookahead_distance:
                return self.path[i]
        return self.path[-1]

    def find_closest_on_path(self, vehicle_position):
        error = np.linalg.norm(self.path - vehicle_position, axis=1)
        idx = np.argmin(error)
        return self.path[idx], error[idx]

    def compute_steering_angle(self, vehicle_position, vehicle_heading):
        self.lookahead_point = self.find_lookahead_point(vehicle_position)
        alpha = np.arctan2(self.lookahead_point[1] - vehicle_position[1],
                           self.lookahead_point[0] - vehicle_position[0]) - vehicle_heading
        steering_angle = np.arctan((2 * self.wheelbase_length * np.sin(alpha)) / self.lookahead_distance)
        
        if abs(steering_angle) >= abs(self.max_steer_angle) and steering_angle > 0:
            return self.max_steer_angle
        elif abs(steering_angle) >= abs(self.max_steer_angle) and steering_angle < 0:
            return -self.max_steer_angle
        else:
            return steering_angle
        
    def curve_length(self, remain_points):
        """Calculate the length of a curve given its points."""
        diffs = np.diff(remain_points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return np.sum(segment_lengths)

    def remaining_curve_length(self, points, current_point_index):
        """Calculate the remaining length of the curve from a given point to the end."""
        remaining_points = points[current_point_index:]
        return self.curve_length(remaining_points)

    def update_path(self, new_path):
        self.path = new_path
        self.path_len = len(new_path)

class GenerateTrajectory:
    def __init__(self, start_point: np.ndarray, end_point: np.ndarray, n_points: int):
        self.x_start, self.y_start = start_point
        self.x_end, self.y_end = end_point
        self.n_points = n_points
        self.x_points = np.linspace(self.x_start, self.x_end, n_points)
        self.y_points = np.linspace(self.y_start, self.y_end, n_points)
    
    def straight_line(self):
        return np.asarray((self.x_points, self.y_points)).T
    
    def sin_wave(self, A=1., omega=1.):
        y = np.sin(self.x_points*omega)*A
        return np.asarray((self.x_points, y)).T
    
    def cosin_wave(self, A=1., omega=1.):
        y = np.cos(self.x_points*omega)*A
        return np.asarray((self.x_points, y)).T
    
    def circle(self, radius, center):
        angles = np.linspace(0, 2 * np.pi, self.n_points)
        x = np.cos(angles)*radius + center[0]
        y = np.sin(angles)*radius + center[1]

        return np.column_stack((x, y))
    
    def generate_n_trajectories(self, n):
        trajectories = []
        for i in range(n):
            traj = np.zeros((self.n_points, 2))
            for j in range(0, 10):
                traj += self.rand_sin_or_cos()
            rand_minus1 = np.random.choice([-1, 1])
            traj[:, 1] *= rand_minus1
            trajectories.append(traj)
        return trajectories
    
    def rand_sin_or_cos(self):
        rand_int = np.random.choice([0, 1])
        rand_amp = np.random.uniform(low=0.5, high=6)
        rand_omega = np.random.uniform(low=0.2, high=10)
        if rand_int:
            return self.sin_wave(A=rand_amp, omega=rand_omega)
        else:
            return self.cosin_wave(A=rand_amp, omega=rand_omega)

    def update_conds(self, start_point: np.ndarray, end_point: np.ndarray, n_points: int):
        self.x_start, self.y_start = start_point
        self.x_end, self.y_end = end_point
        self.x_points = np.linspace(self.x_start, self.x_end, n_points)
        self.y_points = np.linspace(self.y_start, self.y_end, n_points)

class IKine:
    def __init__(self, env):
        self.leg_dict = {
            'fr': self.fr_func,
            'fl': self.fl_func,
            'rr': self.rr_func,
            'rl': self.rl_func
        }
        
        self.leg_list = ["FR", "FL", "RR", "RL"]
        self.leg_id_row = {'FR': 7, 'FL': 10, 'RR': 13, 'RL': 16}
        self.env = env
        self.trunk_com = np.array((3, ))
        self.trunk_rot = np.array((3, 3))
        self.x_des = np.array((3, ))

    def jacobian(self, theta_guess, _1, rot_mat, _3):
        c1, s1, c2, s2, c3, s3 = np.cos(theta_guess[0]), np.sin(theta_guess[0]), \
                                    np.cos(theta_guess[1]), np.sin(theta_guess[1]), \
                                    np.cos(theta_guess[2]), np.sin(theta_guess[2])
        s23, c23 = np.sin(theta_guess[1] + theta_guess[2]), np.cos(theta_guess[1] + theta_guess[2])

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], \
                                                        rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], \
                                                        rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]

        return [[0.213*c1*c2*r12 + 0.213*c1*c23*r12 - 0.08*c1*r13 + 0.213*c2*r13*s1 + 0.213*c23*r13*s1 + 0.08*r12*s1, 0.213*c1*r13*s2 + 0.213*c1*r13*s23 - 0.213*c2*r11 - 0.213*c23*r11 - 0.213*r12*s1*s2 - 0.213*r12*s1*s23, 0.213*c1*r13*s23 - 0.213*c23*r11 - 0.213*r12*s1*s23], 
                [0.213*c1*c2*r22 + 0.213*c1*c23*r22 - 0.08*c1*r23 + 0.213*c2*r23*s1 + 0.213*c23*r23*s1 + 0.08*r22*s1, 0.213*c1*r23*s2 + 0.213*c1*r23*s23 - 0.213*c2*r21 - 0.213*c23*r21 - 0.213*r22*s1*s2 - 0.213*r22*s1*s23, 0.213*c1*r23*s23 - 0.213*c23*r21 - 0.213*r22*s1*s23], 
                [0.213*c1*c2*r32 + 0.213*c1*c23*r32 - 0.08*c1*r33 + 0.213*c2*r33*s1 + 0.213*c23*r33*s1 + 0.08*r32*s1, 0.213*c1*r33*s2 + 0.213*c1*r33*s23 - 0.213*c2*r31 - 0.213*c23*r31 - 0.213*r32*s1*s2 - 0.213*r32*s1*s23, 0.213*c1*r33*s23 - 0.213*c23*r31 - 0.213*r32*s1*s23]
                ]
    
    def jac_inv(self, theta_guess):
        c1, s1, c2, s2, c3, s3 = np.cos(theta_guess[0]), np.sin(theta_guess[0]), \
                                        np.cos(theta_guess[1]), np.sin(theta_guess[1]), \
                                        np.cos(theta_guess[2]), np.sin(theta_guess[2])
        return np.asarray([
            [0, 0.213*s2*s3 + 0.213*c3*c2 + 0.213*c2, -0.213*c2*c3 + 0.213*s3*s2],
            [0.08*s1 + 0.213*c2*c3*c1 - 0.213*c2*c1 - 0.213*c1*s2*s3, -0.213*c3*s1*s2 + 0.213*s1*c2 + 0.213*s1*c2*s3, 0.213*c2*c3*s1 - 0.213*s1*s2*s3],
            [0.213*c1*c2*c3 + 0.213*c1*c2 - 0.213*c1*s2*s3 + 0.08*c1, 0.213*c1*c3*s2 - 0.213*c1*c2 - 0.213*c1*c2*s3, -0.213*c1*c2*c3 + 0.213*c1*s2*s3]
        ]).inv()
    
    def fr_func(self, fr, des, rot_mat, trunk_com):
        c1, s1, c2, s2, c3, s3 = np.cos(fr[0]), np.sin(fr[0]), np.cos(fr[1]), np.sin(fr[1]), np.cos(fr[2]), np.sin(fr[2])
        s23, c23 = np.sin(fr[1] + fr[2]), np.cos(fr[1] + fr[2])

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], \
                                                        rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], \
                                                        rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]

        i_kine_eq = [-0.213*c1*c2*r13 - 0.213*c1*c23*r13 - 0.08*c1*r12 + 0.213*c2*r12*s1 + 0.213*c23*r12*s1 - 0.213*r11*s2 - 0.213*r11*s23 + 0.1881*r11 - 0.04675*r12 - 0.08*r13*s1 + trunk_com[0] - des[0], 
                    -0.213*c1*c2*r23 - 0.213*c1*c23*r23 - 0.08*c1*r22 + 0.213*c2*r22*s1 + 0.213*c23*r22*s1 - 0.213*r21*s2 - 0.213*r21*s23 + 0.1881*r21 - 0.04675*r22 - 0.08*r23*s1 + trunk_com[1] - des[1], 
                    -0.213*c1*c2*r33 - 0.213*c1*c23*r33 - 0.08*c1*r32 + 0.213*c2*r32*s1 + 0.213*c23*r32*s1 - 0.213*r31*s2 - 0.213*r31*s23 + 0.1881*r31 - 0.04675*r32 - 0.08*r33*s1 + trunk_com[2] - des[2]]
        i_kine_eq = np.asarray(i_kine_eq)
        return i_kine_eq
    
    def fl_func(self, fl, des, rot_mat, trunk_com):
        c1, s1, c2, s2, c3, s3 = np.cos(fl[0]), np.sin(fl[0]), np.cos(fl[1]), np.sin(fl[1]), np.cos(fl[2]), np.sin(fl[2])
        s23, c23 = np.sin(fl[1] + fl[2]), np.cos(fl[1] + fl[2])

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], \
                                                        rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], \
                                                        rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]

        i_kine_eq = [0.08*c1*r12 - 0.213*c2*(c1*r13 - r12*s1) - 0.213*c3*(c2*(c1*r13 - r12*s1) + r11*s2) - 0.213*r11*s2 + 0.1881*r11 + 0.04675*r12 + 0.08*r13*s1 - 0.213*s3*(c2*r11 - s2*(c1*r13 - r12*s1)) + trunk_com[0] - des[0],
                    0.08*c1*r22 - 0.213*c2*(c1*r23 - r22*s1) - 0.213*c3*(c2*(c1*r23 - r22*s1) + r21*s2) - 0.213*r21*s2 + 0.1881*r21 + 0.04675*r22 + 0.08*r23*s1 - 0.213*s3*(c2*r21 - s2*(c1*r23 - r22*s1)) + trunk_com[1] - des[1], 
                    0.08*c1*r32 - 0.213*c2*(c1*r33 - r32*s1) - 0.213*c3*(c2*(c1*r33 - r32*s1) + r31*s2) - 0.213*r31*s2 + 0.1881*r31 + 0.04675*r32 + 0.08*r33*s1 - 0.213*s3*(c2*r31 - s2*(c1*r33 - r32*s1)) + trunk_com[2] - des[2]]
        i_kine_eq = np.asarray(i_kine_eq)
        return i_kine_eq

    def rr_func(self, rr, des, rot_mat, trunk_com):
        c1, s1, c2, s2, c3, s3 = np.cos(rr[0]), np.sin(rr[0]), np.cos(rr[1]), np.sin(rr[1]), np.cos(rr[2]), np.sin(rr[2])
        s23, c23 = np.sin(rr[1] + rr[2]), np.cos(rr[1] + rr[2])

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], \
                                                        rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], \
                                                        rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]

        i_kine_eq = [-0.08*c1*r12 - 0.213*c2*(c1*r13 - r12*s1) - 0.213*c3*(c2*(c1*r13 - r12*s1) + r11*s2) - 0.213*r11*s2 - 0.1881*r11 - 0.04675*r12 - 0.08*r13*s1 - 0.213*s3*(c2*r11 - s2*(c1*r13 - r12*s1)) + trunk_com[0] - des[0],
                    -0.08*c1*r22 - 0.213*c2*(c1*r23 - r22*s1) - 0.213*c3*(c2*(c1*r23 - r22*s1) + r21*s2) - 0.213*r21*s2 - 0.1881*r21 - 0.04675*r22 - 0.08*r23*s1 - 0.213*s3*(c2*r21 - s2*(c1*r23 - r22*s1)) + trunk_com[1] - des[1], 
                    -0.08*c1*r32 - 0.213*c2*(c1*r33 - r32*s1) - 0.213*c3*(c2*(c1*r33 - r32*s1) + r31*s2) - 0.213*r31*s2 - 0.1881*r31 - 0.04675*r32 - 0.08*r33*s1 - 0.213*s3*(c2*r31 - s2*(c1*r33 - r32*s1)) + trunk_com[2] - des[2]]
        i_kine_eq = np.asarray(i_kine_eq)
        return i_kine_eq

    def rl_func(self, rl, des, rot_mat, trunk_com):
        c1, s1, c2, s2, c3, s3 = np.cos(rl[0]), np.sin(rl[0]), np.cos(rl[1]), np.sin(rl[1]), np.cos(rl[2]), np.sin(rl[2])
        s23, c23 = np.sin(rl[1] + rl[2]), np.cos(rl[1] + rl[2])

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], \
                                                        rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], \
                                                        rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]

        i_kine_eq = [0.08*c1*r12 - 0.213*c2*(c1*r13 - r12*s1) - 0.213*c3*(c2*(c1*r13 - r12*s1) + r11*s2) - 0.213*r11*s2 - 0.1881*r11 + 0.04675*r12 + 0.08*r13*s1 - 0.213*s3*(c2*r11 - s2*(c1*r13 - r12*s1)) + trunk_com[0] - des[0],
                    0.08*c1*r22 - 0.213*c2*(c1*r23 - r22*s1) - 0.213*c3*(c2*(c1*r23 - r22*s1) + r21*s2) - 0.213*r21*s2 - 0.1881*r21 + 0.04675*r22 + 0.08*r23*s1 - 0.213*s3*(c2*r21 - s2*(c1*r23 - r22*s1)) + trunk_com[1] - des[1], 
                    0.08*c1*r32 - 0.213*c2*(c1*r33 - r32*s1) - 0.213*c3*(c2*(c1*r33 - r32*s1) + r31*s2) - 0.213*r31*s2 - 0.1881*r31 + 0.04675*r32 + 0.08*r33*s1 - 0.213*s3*(c2*r31 - s2*(c1*r33 - r32*s1)) + trunk_com[2] - des[2]]
        i_kine_eq = np.asarray(i_kine_eq)
        return i_kine_eq

    def velocity(self, joints, des, rot_mat, leg=None):
        c1, s1, c2, s2, c3, s3 = np.cos(joints[0]), np.sin(joints[0]), np.cos(joints[1]), np.sin(joints[1]), np.cos(joints[2]), np.sin(joints[2])
        s23, c23 = np.sin(joints[1] + joints[2]), np.cos(joints[1] + joints[2])

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], \
                                                        rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], \
                                                        rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2]
        
        d_t1, d_t2, d_t3 = des[0], des[1], des[2]
        
        if leg == 'rr':
            vel = np.asarray([0.213*c1*c2*d_t1*r12 + 0.213*c1*c23*d_t1*r12 - 0.08*c1*d_t1*r13 + 0.213*c1*d_t2*r13*s2 + 0.213*c1*r13*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r13*s1 - 0.213*c2*d_t2*r11 + 0.213*c23*d_t1*r13*s1 - 0.213*c23*r11*(d_t2 + d_t3) + 0.08*d_t1*r12*s1 - 0.213*d_t2*r12*s1*s2 - 0.213*r12*s1*s23*(d_t2 + d_t3),
                                0.213*c1*c2*d_t1*r22 + 0.213*c1*c23*d_t1*r22 - 0.08*c1*d_t1*r23 + 0.213*c1*d_t2*r23*s2 + 0.213*c1*r23*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r23*s1 - 0.213*c2*d_t2*r21 + 0.213*c23*d_t1*r23*s1 - 0.213*c23*r21*(d_t2 + d_t3) + 0.08*d_t1*r22*s1 - 0.213*d_t2*r22*s1*s2 - 0.213*r22*s1*s23*(d_t2 + d_t3),
                                0.213*c1*c2*d_t1*r32 + 0.213*c1*c23*d_t1*r32 - 0.08*c1*d_t1*r33 + 0.213*c1*d_t2*r33*s2 + 0.213*c1*r33*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r33*s1 - 0.213*c2*d_t2*r31 + 0.213*c23*d_t1*r33*s1 - 0.213*c23*r31*(d_t2 + d_t3) + 0.08*d_t1*r32*s1 - 0.213*d_t2*r32*s1*s2 - 0.213*r32*s1*s23*(d_t2 + d_t3)])
        if leg == 'fr':
            vel = np.asarray([0.213*c1*c2*d_t1*r12 + 0.213*c1*c23*d_t1*r12 - 0.08*c1*d_t1*r13 + 0.213*c1*d_t2*r13*s2 + 0.213*c1*r13*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r13*s1 - 0.213*c2*d_t2*r11 + 0.213*c23*d_t1*r13*s1 - 0.213*c23*r11*(d_t2 + d_t3) + 0.08*d_t1*r12*s1 - 0.213*d_t2*r12*s1*s2 - 0.213*r12*s1*s23*(d_t2 + d_t3),
                                0.213*c1*c2*d_t1*r22 + 0.213*c1*c23*d_t1*r22 - 0.08*c1*d_t1*r23 + 0.213*c1*d_t2*r23*s2 + 0.213*c1*r23*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r23*s1 - 0.213*c2*d_t2*r21 + 0.213*c23*d_t1*r23*s1 - 0.213*c23*r21*(d_t2 + d_t3) + 0.08*d_t1*r22*s1 - 0.213*d_t2*r22*s1*s2 - 0.213*r22*s1*s23*(d_t2 + d_t3),
                                0.213*c1*c2*d_t1*r32 + 0.213*c1*c23*d_t1*r32 - 0.08*c1*d_t1*r33 + 0.213*c1*d_t2*r33*s2 + 0.213*c1*r33*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r33*s1 - 0.213*c2*d_t2*r31 + 0.213*c23*d_t1*r33*s1 - 0.213*c23*r31*(d_t2 + d_t3) + 0.08*d_t1*r32*s1 - 0.213*d_t2*r32*s1*s2 - 0.213*r32*s1*s23*(d_t2 + d_t3)])
        if leg == 'fl':
            vel = np.asarray([0.213*c1*c2*d_t1*r12 + 0.213*c1*c23*d_t1*r12 - 0.08*c1*d_t1*r13 + 0.213*c1*d_t2*r13*s2 + 0.213*c1*r13*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r13*s1 - 0.213*c2*d_t2*r11 + 0.213*c23*d_t1*r13*s1 - 0.213*c23*r11*(d_t2 + d_t3) + 0.08*d_t1*r12*s1 - 0.213*d_t2*r12*s1*s2 - 0.213*r12*s1*s23*(d_t2 + d_t3),
                                0.213*c1*c2*d_t1*r22 + 0.213*c1*c23*d_t1*r22 - 0.08*c1*d_t1*r23 + 0.213*c1*d_t2*r23*s2 + 0.213*c1*r23*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r23*s1 - 0.213*c2*d_t2*r21 + 0.213*c23*d_t1*r23*s1 - 0.213*c23*r21*(d_t2 + d_t3) + 0.08*d_t1*r22*s1 - 0.213*d_t2*r22*s1*s2 - 0.213*r22*s1*s23*(d_t2 + d_t3),
                                0.213*c1*c2*d_t1*r32 + 0.213*c1*c23*d_t1*r32 - 0.08*c1*d_t1*r33 + 0.213*c1*d_t2*r33*s2 + 0.213*c1*r33*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r33*s1 - 0.213*c2*d_t2*r31 + 0.213*c23*d_t1*r33*s1 - 0.213*c23*r31*(d_t2 + d_t3) + 0.08*d_t1*r32*s1 - 0.213*d_t2*r32*s1*s2 - 0.213*r32*s1*s23*(d_t2 + d_t3)])
        if leg == 'rl':
            vel = np.asarray([0.213*c1*c2*d_t1*r12 + 0.213*c1*c23*d_t1*r12 - 0.08*c1*d_t1*r13 + 0.213*c1*d_t2*r13*s2 + 0.213*c1*r13*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r13*s1 - 0.213*c2*d_t2*r11 + 0.213*c23*d_t1*r13*s1 - 0.213*c23*r11*(d_t2 + d_t3) + 0.08*d_t1*r12*s1 - 0.213*d_t2*r12*s1*s2 - 0.213*r12*s1*s23*(d_t2 + d_t3),
                                0.213*c1*c2*d_t1*r22 + 0.213*c1*c23*d_t1*r22 - 0.08*c1*d_t1*r23 + 0.213*c1*d_t2*r23*s2 + 0.213*c1*r23*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r23*s1 - 0.213*c2*d_t2*r21 + 0.213*c23*d_t1*r23*s1 - 0.213*c23*r21*(d_t2 + d_t3) + 0.08*d_t1*r22*s1 - 0.213*d_t2*r22*s1*s2 - 0.213*r22*s1*s23*(d_t2 + d_t3),
                                0.213*c1*c2*d_t1*r32 + 0.213*c1*c23*d_t1*r32 - 0.08*c1*d_t1*r33 + 0.213*c1*d_t2*r33*s2 + 0.213*c1*r33*s23*(d_t2 + d_t3) + 0.213*c2*d_t1*r33*s1 - 0.213*c2*d_t2*r31 + 0.213*c23*d_t1*r33*s1 - 0.213*c23*r31*(d_t2 + d_t3) + 0.08*d_t1*r32*s1 - 0.213*d_t2*r32*s1*s2 - 0.213*r32*s1*s23*(d_t2 + d_t3)])
        return self.jac_inv @ vel

    def choose_leg(self, leg_id):
        if leg_id in self.leg_dict:
            return self.leg_dict[leg_id]

    def get_trunk_loc(self, leg):
        return quat2rot_mj(np.asarray(self.env.robot_data.qpos[3:7])).T, self.env.robot_data.qpos[:3]

    # def solve(self, leg=None, leg_idx=None, theta=None, x_des=None):
    #     trunk_rot, trunk_com = self.get_trunk_loc(leg)
    #     sol = self.solve_eq(trunk_rot, trunk_com, leg, leg_idx, theta, x_des)
    #     self.env.check_joint_angles(leg_idx=leg_idx, result=sol)

    #     return sol

    # def solve_eq(self, trunk_rot, trunk_com, leg=None, leg_idx=None, theta=None, x_des=None):
    #     leg_idx -= 7 # Subtracting 7 and the ranges vector starts from 0, as opposed to the qpos vector that starts at idx 7.
    #     # trunk_rot, trunk_com = self.get_trunk_loc(leg)

    #     result = optimize.root(self.choose_leg(leg_id=leg), x0=theta, args=(x_des, trunk_rot, trunk_com), jac=self.jacobian, method='hybr', tol=1e-6)
    #     self.env.check_joint_angles(leg_idx=leg_idx, result=result.x)

    #     return result.x

    def solve(self, leg=None, leg_idx=None, theta=None, x_des=None):
        leg_idx -= 7 # Subtracting 7 and the ranges vector starts from 0, as opposed to the qpos vector that starts at idx 7.
        trunk_rot, trunk_com = self.get_trunk_loc(leg)
        result = optimize.root(self.choose_leg(leg_id=leg), x0=theta, args=(x_des, trunk_rot, trunk_com), jac=self.jacobian, method='hybr', tol=1e-6)
        self.env.check_joint_angles(leg_idx=leg_idx, result=result.x)
        
        return result.x

    # @timer
    # @jit
    def get_angles(self, com_des_arr, des_array, des_angles=None):
        if des_angles is None:
            des_angles = [0, 0, 0]
        fr_guess = copy.deepcopy(self.env.robot_data.qpos[7:10])
        fl_guess = copy.deepcopy(self.env.robot_data.qpos[10:13])
        rr_guess = copy.deepcopy(self.env.robot_data.qpos[13:16])
        rl_guess = copy.deepcopy(self.env.robot_data.qpos[16:19])
        
        guess_array = [fr_guess, fl_guess, rr_guess, rl_guess]
        now_ort = quat2eul(self.env.robot_data.qpos[3:7])

        del fr_guess, fl_guess, rr_guess, rl_guess

        ### New Try:
        now_ort_quat = Quaternion(self.env.robot_data.qpos[3:7])
        des_quat = Quaternion(eul2quat(des_angles))
        q_diff = now_ort_quat.inverse * des_quat
        des_eul = quat2eul(q_diff.q)

        now_x_angle = now_ort[0]
        now_y_angle = now_ort[1]
        now_z_angle = now_ort[2]

        delta_x_angle = des_eul[0]
        delta_y_angle = des_eul[1]
        delta_z_angle = des_eul[2]

        # com_des_arr = com_des_arr.reshape((1, 3))
        arr_length = len(com_des_arr)
        des_trunk_x_angles = np.linspace(now_x_angle, now_x_angle+delta_x_angle, arr_length)
        des_trunk_y_angles = np.linspace(now_y_angle, now_y_angle+delta_y_angle, arr_length)
        des_trunk_z_angles = np.linspace(now_z_angle, now_z_angle+delta_z_angle, arr_length)
        des_trunk_angles = np.asarray([des_trunk_x_angles, des_trunk_y_angles, des_trunk_z_angles]).T
        
        # tot_traj = []
        trunk_angles = eul2quat(des_trunk_angles)
        tot_traj = [None] * len(com_des_arr)
        leg_list_local = self.leg_list

        # def process_outer_loop(idx, com_des, current_ort):
        #     angles = []
        #     trunk_rot, trunk_com = quat2rot_mj(current_ort).T, com_des

        #     for idx_, [leg, des, gss] in enumerate(zip(leg_list_local, des_array, guess_array)):
        #         angles.extend(optimize.root(fun=self.choose_leg(leg_id=leg.lower()), x0=gss, args=(des, trunk_rot, trunk_com), jac=self.jacobian, method='hybr', tol=1e-6).x)
        #         guess_array[idx_][:] = angles[-3:]

        #     tot_traj[idx] = angles

        def process_outer_loop(idx, com_des, current_ort):
            angles = []
            trunk_rot, trunk_com = quat2rot_mj(current_ort).T, com_des

            for idx_, [leg, des, gss] in enumerate(zip(leg_list_local, des_array, guess_array)):
                angles.extend(optimize.root(fun=self.choose_leg(leg_id=leg.lower()), x0=gss, args=(des, trunk_rot, trunk_com), jac=self.jacobian, method='hybr', tol=1e-6).x)
                guess_array[idx_][:] = angles[-3:]

            tot_traj[idx] = angles

        with ThreadPoolExecutor(max_workers=4) as executor:
            for _ in executor.map(process_outer_loop, range(len(com_des_arr)), com_des_arr, trunk_angles):
            # for _ in executor.map(process_outer_loop, range(len(com_des_arr)), com_des_arr.reshape((1, 3)), trunk_angles):
                pass

        return tot_traj, np.asarray(trunk_angles)

        # for idx, com_des in enumerate(com_des_arr):
        #     current_ort = trunk_angles[idx]
        #     angles = []
        #     trunk_rot, trunk_com = quat2rot(current_ort).T, com_des

        #     for idx_, [leg, des, gss] in enumerate(zip(self.leg_list, des_array, guess_array)):
        #         angles.extend(optimize.root(fun=self.choose_leg(leg_id=leg.lower()), x0=gss, args=(des, trunk_rot, trunk_com), jac=self.jacobian, method='hybr', tol=1e-6).x)
        #         guess_array[idx_][:] = angles[-3:]
            
        #     tot_traj.append(angles)

        # return tot_traj, np.asarray(trunk_angles)

class OUActionNoise(object):
    """
    OU - Orenstein - Ulenbeck Noise. Type of noise that models the motion of a brownian particle.
    Temporarly corelated, which means corelated in time type of noise, centers around a mean of 0.
    """
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # noise = OUActionNoise()
        # noise() 
        # # --> Just use noise() instead of ObjName.noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class MakeVideo:
    def __init__(self, env, framerate=60, cam_dist=2, width=640, height=480):
        self.env = env
        self.frames = []
        self.renderer = mj.Renderer(env.robot, width=width, height=height)
        self.framerate = framerate
        self.camera = mj.MjvCamera()
        self.camera.distance = cam_dist
    
    def capture(self, frame, look_at_body='trunk'):
        self.camera.lookat = self.env.robot_data.body(look_at_body).subtree_com
        time = self.env.robot_data.time*1e5*0.1
        self.camera.azimuth = self.azimuth(time=time, degrees=360)
        self.camera.elevation = self.altitude(time, degrees=5)
        self.renderer.update_scene(frame, self.camera)
        pixels = self.renderer.render()
        self.frames.append(pixels)
    
    def get_frames(self):
        return np.asarray(self.frames)

    def unit_smooth(self, normalised_time: float) -> float:
        return 1 - np.cos(normalised_time*0.55*np.pi)
    def azimuth(self, time: float, degrees: float) -> float:
        return 100 + self.unit_smooth(time/600) * degrees
    def altitude(self, time: float, degrees: float) -> float:
        return -25 + self.unit_smooth(time/300) * degrees

class Go1Env():
    def __init__(self, xml, env_num):
        mujoco_model = mj.MjModel.from_xml_path(xml)
        
        self.xml_model = mujoco_model
        self.tree = ET.parse(xml)
        self.root = self.tree.getroot()
        self.robot = mujoco_model
        self.robot_data = mj.MjData(self.robot)
        self.dt = self.robot.opt.timestep
        
        self.get_bodies_geom_nums()
        self.fr_calf_num = self.geom_name_to_body_name['FR_calf'][0]
        self.fl_calf_num = self.geom_name_to_body_name['FL_calf'][0]
        self.rr_calf_num = self.geom_name_to_body_name['RR_calf'][0]
        self.rl_calf_num = self.geom_name_to_body_name['RL_calf'][0]
        self.paws_dict = {'fr': self.fr_calf_num, 'fl': self.fl_calf_num, 'rr': self.rr_calf_num, 'rl': self.rl_calf_num}
        self.env_num = env_num

        self.DoF_num = self.robot.nu
        self.DoF_names = [self.robot.joint(i).name for i in range(self.robot.njnt)][1:]
        self.bodies_names = [self.robot.body(i).name for i in range(self.robot.nbody)][1:]
        self.joint_ranges = self.robot.jnt_range[1:]
        self.contact_ids = ()
        self.paws_ids = [4, 10, 7, 13]
        self.legs_ids = ['FR', 'RR', 'FL', 'RL']
        self.force_sensors_id = [mj.mj_name2id(self.robot, mj.mjtObj.mjOBJ_SENSOR, id+'_force') for id in self.legs_ids]
        self.geom_group_set = np.zeros(6, ) + [1, 0, 0, 0, 0, 0]
        self.geom_group_set = self.geom_group_set.astype(np.uint8)
        self.go1_contacts = set([])

        self.L1 = 0.08
        self.L2 = 0.213
        self.L3 = 0.213
        self.L2sq = 0.213**2
        self.L3sq = 0.213**2

        self.step_cnt = 0
        self.first_render = 0
        self.paws_on_ground = 0
        self.step_count = 0
        self.reward_components = np.zeros((1, 13))
        self.paws_done = False
        self.flipped_done = False
        self.z_force_margin = -0.1
        # ### Done and reset terms:

        self.joint_not_in_range = False
        self.done = False

        self.A = np.zeros((2*self.robot.nv, 2*self.robot.nv))
        self.B = np.zeros((2*self.robot.nv, self.DoF_num))
        self.C = np.zeros((self.robot.nsensordata, 2*self.robot.nv))
        self.D = np.zeros((self.robot.nsensordata, self.DoF_num))
        
        self.stochastic_start = np.random.randn((2)) * 0.1
        # self.init_pos = np.array([0.1+self.stochastic_start[0],  0.35+self.stochastic_start[1],  0.46189675, \
        self.init_pos = np.array([0.1+self.stochastic_start[0],  0.35+self.stochastic_start[1],  1.23189675, \
        # self.init_pos = np.array([0.1,  0.35,  1.28189675, \
                                        1.0, 0.0, 0.0, 0.0, \
                                        -0.1, 0.42965716e+00, -1.0, \
                                        0.1,  0.42965716e+00, -1.0, \
                                        -0.1,  0.42965716e+00, -1.0, \
                                        0.1,  0.42965716e+00, -1.0]) #"""For Height Map"""

        # self.init_pos = np.array([0.0,  0.0,  0.45189675, \
        #                                 1.0, 0.0, 0.0, 0.0, \
        #                                 -0.1, 0.42965716e+00, -1.0, \
        #                                 0.1,  0.42965716e+00, -1.0, \
        #                                 -0.1,  0.42965716e+00, -1.0, \
        #                                 0.1,  0.42965716e+00, -1.0]) #"""For Stairs Map"""
        
        self.init_state_space = self.get_state_space()
        self.init_ctrl = self.get_init_ctrl()
        _, _, self.touching_names = self.count_touching_paws()

        self.noise = OUActionNoise(mu=np.zeros(12))

    def get_bodies_geom_nums(self):
        self.geom_name_to_body_name = {}
        self.geom_name_to_body_name['hfield1'] = [0]
        for idx, body in enumerate(self.root.iter('body')):
            for idx, geom in enumerate(self.root.iter('geom')):
                if geom.get('name') is None:
                    continue
                if body.get('name') in self.geom_name_to_body_name and isinstance(self.geom_name_to_body_name[body.get('name')], list) and (body.get('name') in geom.get('name')):
                    self.geom_name_to_body_name[body.get('name')].append(self.get_geom_id(geom_name=geom.get('name')))
                elif (body.get('name') not in geom.get('name')):
                    continue
                else:
                    self.geom_name_to_body_name[body.get('name')] = [self.get_geom_id(geom_name=geom.get('name'))]

    def get_body_id(self, body_name):
        return self.robot.body(body_name).id

    def get_body_name(self, body_id):
        return mj.mj_id2name(self.robot, mj.mjtObj.mjOBJ_BODY, body_id)

    def get_joint_id(self, joint_name):
        return self.robot.joint(joint_name).id
    
    def get_site_id(self, site_name):
        return self.robot.site(site_name).id
    
    def get_geom_id(self, geom_name):
        try:
            return self.robot.geom(geom_name).id
        except:
            return None

    def get_geom_name(self, geom_id):
        return mj.mj_id2name(self.robot, mj.mjtObj.mjOBJ_GEOM, geom_id)

    def get_body_pos(self, body_name):
        return self.robot_data.xpos[self.get_body_id(body_name)]
    
    def get_joint_anchor(self, joint_name):
        return self.robot_data.xanchor[self.get_joint_id(joint_name)]

    def get_site_pos(self, site_name):
        return self.robot_data.site_xpos[self.get_site_id(site_name)]
    
    def get_body_quat(self, body_name):
        return self.robot_data.xquat[self.get_body_id(body_name)]
    
    def get_body_rot(self, body_name):
        return quat2rot_mj(self.get_body_quat(body_name))

    def get_body_eul(self, body_name):
        eul_angles = quat2eul(self.get_body_quat(body_name))

        return eul_angles

    def get_com_z_dist(self, data, point_from):
        geom_id_arr = np.zeros((1, 1)).astype(np.int32)
        trunk_num = self.get_body_id(body_name='trunk')
        # z_dist = mj.mj_ray(self.robot, data, point_from, np.asarray([0, 0, -1]), None, 1, -1, geom_id_arr)
        z_dist = mj.mj_ray(self.robot, data, point_from.reshape(3, 1), np.asarray([0, 0, -1]).reshape(3, 1), self.geom_group_set.reshape(-1, 1), 1, 1, geom_id_arr)

        return z_dist, geom_id_arr
    
    def get_com_z_dist_hfield(self, data, point_from):
        z_dist = mj.mj_rayHfield(self.robot, data, self.get_geom_id("hfield1"), 
                              point_from, np.asarray([0, 0, -1]))
        return z_dist
    
    def get_total_com(self):
        return self.robot_data.subtree_com

    def get_torque(self, xd, data_copy):
        data_copy.qpos = xd[:]
        data_copy.qacc = 0
        mj.mj_inverse(self.robot, data_copy)
        force = data_copy.qfrc_inverse.copy()
        torque = np.atleast_2d(force) @ np.linalg.pinv(data_copy.actuator_moment)

        return torque

    def compute_zmp_world_coords(self, force_vectors):
        # paws_world = np.array(paw_coordinates)
        paws_world = self.get_4_paws()
        forces = np.array(force_vectors)
        com_world = self.robot_data.qpos[:3]
        
        # Translate paw coordinates to be relative to CoM
        paws_relative = paws_world - com_world
        
        # Compute moments using vectorized operations
        tau_x = np.sum(forces[:, 1] * paws_relative[:, 2] - forces[:, 2] * paws_relative[:, 1])
        tau_y = np.sum(forces[:, 2] * paws_relative[:, 0] - forces[:, 0] * paws_relative[:, 2])
        F_z_total = np.sum(forces[:, 2])
        
        # Calculate ZMP
        x_zmp = tau_y / F_z_total
        y_zmp = tau_x / F_z_total
        
        return x_zmp, y_zmp

    def get_touching_pairs(self):
        fr_touch = 0
        fl_touch = 0
        rr_touch = 0
        rl_touch = 0
        touch_list = []
        geoms1 = self.robot_data.contact.geom1
        geoms2 = self.robot_data.contact.geom2
        try:
            for idx, geom in enumerate(geoms2):
                if geom == self.fr_calf_num and (geoms1[idx] == 0 or geoms1[idx] == 1 or geoms1[idx] == 2 or geoms1[idx] == 3):
                    fr_touch = 1
                    touch_list.append("FR")
                    continue
                
                if geom == self.fl_calf_num and (geoms1[idx] == 0 or geoms1[idx] == 1 or geoms1[idx] == 2 or geoms1[idx] == 3):
                    fl_touch = 1
                    touch_list.append("FL")
                    continue          
                
                if geom == self.rr_calf_num and (geoms1[idx] == 0 or geoms1[idx] == 1 or geoms1[idx] == 2 or geoms1[idx] == 3):
                    rr_touch = 1
                    touch_list.append("RR")
                    continue               

                if geom == self.rl_calf_num and (geoms1[idx] == 0 or geoms1[idx] == 1 or geoms1[idx] == 2 or geoms1[idx] == 3):
                    rl_touch = 1
                    touch_list.append("RL")
                    continue

            return np.array([fr_touch, fl_touch, rr_touch, rl_touch]), touch_list
        except:
            return np.zeros((0, 4)), touch_list

    def get_touching_parts(self):
        contacts = self.robot_data.contact
        world_contact_indices = np.where(contacts.geom1==0)
        self.go1_contacts = set(contacts.geom2[world_contact_indices])
        names = [self.get_geom_name(geom_id) for geom_id in self.go1_contacts]
        # TODO: Continue forces analysis!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return names, self.go1_contacts

    def get_force_sensors(self):
        rl_force = self.robot_data.sensordata[-3:]
        rr_force = self.robot_data.sensordata[-6:-3]
        fl_force = self.robot_data.sensordata[-9:-6]
        fr_force = self.robot_data.sensordata[-12:-9]
        forces = np.asarray([fr_force, fl_force, rr_force, rl_force])
        # mj.mj_inverse(self.robot, self.robot_data)
        # forces_ = self.robot_data.qfrc_inverse
        return forces

    def get_contact_forces(self):
        contacts = self.robot_data.contact
        self.contact_ids = contacts.efc_address
        if len(contacts.geom2) >= 1:
            ids_12 = np.where(contacts.geom2==self.fr_calf_num)[0]
            ids_21 = np.where(contacts.geom2==self.fl_calf_num)[0]
            ids_30 = np.where(contacts.geom2==self.rr_calf_num)[0]
            ids_39 = np.where(contacts.geom2==self.rl_calf_num)[0]

        try:
            fr_force = np.zeros((len(ids_12), 6))
            for i, id in enumerate(ids_12):
                mj.mj_contactForce(self.robot, self.robot_data, id, fr_force[i])
                mj.mju_rotVecMatT(fr_force[i, :3], fr_force[i, :3], contacts.frame[i])
            fr_force = np.sum(fr_force, axis=0)[0:3]
        except:
            fr_force = np.zeros(3,)
        
        try:
            fl_force = np.zeros((len(ids_21), 6))
            for i, id in enumerate(ids_21):
                mj.mj_contactForce(self.robot, self.robot_data, id, fl_force[i])
                mj.mju_rotVecMatT(fl_force[i, :3], fl_force[i, :3], contacts.frame[i])
            fl_force = np.sum(fl_force, axis=0)[0:3]
        except:
            fl_force = np.zeros(3,)

        try:
            rr_force = np.zeros((len(ids_30), 6))
            for i, id in enumerate(ids_30):
                mj.mj_contactForce(self.robot, self.robot_data, id, rr_force[i])
                mj.mju_rotVecMatT(rr_force[i, :3], rr_force[i, :3], contacts.frame[i])
            rr_force = np.sum(rr_force, axis=0)[0:3]
        except:
            rr_force = np.zeros(3,)

        try:
            rl_force = np.zeros((len(ids_39), 6))
            for i, id in enumerate(ids_39):
                mj.mj_contactForce(self.robot, self.robot_data, id, rl_force[i])
                mj.mju_rotVecMatT(rl_force[i, :3], rl_force[i, :3], contacts.frame[i])
            rl_force = np.sum(rl_force, axis=0)[0:3]
        except:
            rl_force = np.zeros(3,)

        forces = np.stack((fr_force, fl_force, rr_force, rl_force), axis=1).T
        return forces

    def get_paw_contact_force(self, leg_id):
        contacts = self.robot_data.contact
        contacts_nums = np.where(contacts.geom2==self.paws_dict[leg_id.lower()])[0]
        try:
            force = np.zeros((len(contacts_nums), 6))
            normal = np.zeros((len(contacts_nums), 3))
            for i, id in enumerate(contacts_nums):
                mj.mj_contactForce(self.robot, self.robot_data, id, force[i])
                mj.mju_rotVecMatT(force[i, :3], force[i, :3], contacts.frame[id])
                normal += contacts.frame[id][:3]
            force = np.sum(force, axis=0)[0:3]
            normal = np.sum(normal, axis=0)[0:3]
        except:
            force = np.zeros(3,)
            normal = np.zeros(3,)
        
        return force, normal/np.linalg.norm(normal)

    def check_force_direction(self, force):
        # ### Return True if needs to make another step, False otherwise
        x, y, z = abs(force[0]), abs(force[1]), abs(force[2])
        if y > (z - self.z_force_margin):
            return True
        elif x > (z - self.z_force_margin):
            return True
        else:
            return False 

    def local2global(self, body_name, point_on_body):
        global_point = self.get_joint_anchor(body_name+"_joint") + quat2rot_mj(self.get_body_quat(body_name)) @ np.array(point_on_body)
        
        return global_point
    
    def local2global_mj(self, body_name, point_on_body):
        rot_mat = np.zeros((9, 1))
        global_point = np.zeros((3, 1))
        mj.mj_local2Global(self.robot_data, global_point, rot_mat, np.array(point_on_body), self.get_body_quat(body_name), self.get_body_id(body_name), False)
        rot_mat = rot_mat.reshape((3, 3))

        return rot_mat, global_point.reshape((3, ))

    def global2local(self, body_name, point):
        body_pos = self.get_body_pos(body_name)
        rot_mat = quat2rot_mj(self.get_body_quat(body_name))
        local_point = rot_mat.transpose() @ (np.array(point) - body_pos)

        return local_point
    
    def get_state_space(self):
        self.u = self.robot_data.ctrl[:]
        self.x = self.robot_data.qpos[:]
        self.x_dot = self.robot_data.qvel[:]

        mj.mjd_transitionFD(self.robot, self.robot_data, 1e-6, True, self.A, self.B, None, None)
        full_state_space = [self.x, self.x_dot, self.u, self.A, self.B]

        return full_state_space

    def get_init_ctrl(self):
        self.robot_data.qpos = self.init_pos
        mj.mj_forward(self.robot, self.robot_data)
        self.robot_data.qacc = 0
        mj.mj_inverse(self.robot, self.robot_data)
        qfrc0 = self.robot_data.qfrc_inverse.copy()
        ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(self.robot_data.actuator_moment)
        ctrl0 = ctrl0.flatten()
        return ctrl0

    def get_com_jac(self):
        mj.mj_forward(self.robot, self.robot_data)
        jac_com = np.zeros((3, self.robot.nv))
        mj.mj_jacSubtreeCom(self.robot, self.robot_data, jac_com, self.get_body_id("trunk"))

        return jac_com

    def get_body_jac(self, body_name):
        mj.mj_forward(self.robot, self.robot_data)
        jac_body = np.zeros((3, self.robot.nv))
        mj.mj_jacBodyCom(self.robot, self.robot_data, jac_body, None, self.get_body_id(body_name))

        return jac_body

    def get_support_polygon_com(self, stepping_leg):
        legs = ['FR', 'RL', 'FL', 'RR']
        # touching_paws = self.touching_names.copy()
        touching_paws = legs.copy()
        try:
            touching_paws.remove(stepping_leg)
        except:
            print("Check Paws!!")
            return np.zeros((3, ))
        pos = np.zeros((3, ))
        for paw in touching_paws:
            _ ,pos_i = self.local2global_mj(paw+"_calf", [0, 0, -0.213])
            pos += pos_i

        return pos/3
    
    def get_polygon_line(self, stepping_leg, n=500, z_des=0.2250):
        polygon_com = self.get_support_polygon_com(stepping_leg)
        # com_tot = self.get_body_pos(body_name="trunk")
        com_tot = self.get_total_com()[0]
        # print('Trunk Com: ', com)
        # print('Total Com: ', com_tot)

        polygon_com[-1] = com_tot[-1]
        temp = self.robot_data.qpos[0:3].copy()
        self.robot_data.qpos[0:3] = polygon_com[:]

        # head_point = self.get_site_pos(site_name="LiDAR")
        head_point = com_tot
        # head_point = copy.deepcopy(self.robot_data.qpos[0:3]) + [0.1, 0, 0]
        
        # ### For kine:
        # z_dist, geom_intersect = self.get_com_z_dist(data=data_copy, point_from=head_point)
        z_dist, geom_intersect = self.get_com_z_dist(data=self.robot_data, point_from=head_point)
        # z_dist = self.get_com_z_dist_hfield(data=data_copy, point_from=head_point)

        self.robot_data.qpos[0:3] = temp[:]
        del temp
        z_height = z_dist - z_des

        line_ = line(x0=com_tot[0], y0=com_tot[1], x1=polygon_com[0], y1=polygon_com[1], m=0, 
                        n=n, z=com_tot[-1], z_end = com_tot[-1]-z_height)

        return line_

    def get_qpos_without_quat(self, ):
        dq = np.zeros(self.robot.nv)
        dq[0:3] = self.robot_data.qpos[0:3]
        dq[3:6] = quat2eul(self.robot_data.qpos[3:7])
        dq[6:] = self.robot_data.qpos[7:]
        # mj.mj_differentiatePos(self.robot, dq, 1, now_pos*0, now_pos)
        return dq

    def get_4_paws(self):
        return np.asarray((self.get_site_pos('FR_touch_sensor'), 
                           self.get_site_pos('FL_touch_sensor'), 
                           self.get_site_pos('RR_touch_sensor'), 
                           self.get_site_pos('RL_touch_sensor'),)).reshape(12, )

    def step_sim(self, action=None):
        if action is None:
            action = np.zeros((12, ))

        self.robot_data.ctrl = np.array(action)
        
        self.step_count += 1
        self.step_cnt += 1

        self.mj_step()
        self.check_if_dones()

        # return self.state, reward, self.done, self.info
        return self.done
    
    def mj_step(self):
        mj.mj_step(self.robot, self.robot_data)

    def check_if_dones(self):
        if self.joint_not_in_range:
            # print('Joint not in range. Reset environment.')
            self.done = True
        else:
            self.check_trunk_direction()
            
    def check_joint_angles(self, leg_idx, result):
        is_in_range = (self.joint_ranges[leg_idx:leg_idx+3, 0] < result) & (result < self.joint_ranges[leg_idx:leg_idx+3, 1])
        
        if not any(is_in_range):
            # print('Leg IDX: ', leg_idx, 'Joint Range: ', self.joint_ranges[leg_idx:leg_idx+3])
            # print("Values: ", result)
            # print(is_in_range)
            self.joint_not_in_range = True

    def count_touching_paws(self):
        touching_idx, touching_names = self.get_touching_pairs()
        cnt = touching_idx.sum()
        return cnt, set(touching_idx), set(touching_names)
    
    def paws_force(self, paws_force_weight=0.001):
        forces = np.array(self.observation_dict['Paws']) * paws_force_weight

        return forces

    def check_trunk_direction(self):
        direc = self.get_body_rot(body_name='trunk')[:, -1]
        if direc[-1] < 0.5:
            # print('Robot in wrong orientation. Reset environment.')
            self.done = True

    def check_paws_location(self, z_new):
        _, fr_paw = self.local2global_mj(body_name='FR_calf', point_on_body=[0, 0, -0.213])
        _, fl_paw = self.local2global_mj(body_name='FL_calf', point_on_body=[0, 0, -0.213])
        _, rr_paw = self.local2global_mj(body_name='RR_calf', point_on_body=[0, 0, -0.213])
        _, rl_paw = self.local2global_mj(body_name='RL_calf', point_on_body=[0, 0, -0.213])

        fr_joint = self.get_body_pos(body_name='FR_calf')
        fl_joint = self.get_body_pos(body_name='FL_calf')
        rr_joint = self.get_body_pos(body_name='RR_calf')
        rl_joint = self.get_body_pos(body_name='RL_calf')

        pos_diff = np.array([fr_joint[2] - fr_paw[2], fl_joint[2] - fl_paw[2], rr_joint[2] - rr_paw[2], rl_joint[2] - rl_paw[2]])
        paw_vs_com = np.array([z_new - fr_paw[2], z_new - fl_paw[2], z_new - rr_paw[2], z_new - rl_paw[2]])
        pos_bool = (pos_diff > 0).astype(int)
        paw_vs_com_bool = (paw_vs_com > 0).astype(int)
        
        pos_bool = np.array([pos_bool[i] if i == 1 else (pos_bool[i]*-1) for i in paw_vs_com_bool])
        pos_diff = np.array([pos_diff[i] if i == 1 else (pos_diff[i]*-1) for i in paw_vs_com_bool])
        return pos_bool, pos_diff

    def write_state(self, state_file, state, angles):
        with open(file=state_file, mode='a') as f:
            f.write(str(state)+"\n"+str(angles)+"\n")

    def reset_sim(self, init_pos, booted=False):
        # mj.mj_resetData(self.robot, mj.MjData(self.robot))
        # mj.mj_resetDataKeyframe(self.robot, self.robot_data, 1)
        # self.robot_data = mj.MjData(self.xml_model)
        # self.robot_data.qpos = [*self.init_pos[:]]
        
        self.robot_data.qpos = [*init_pos[:]]
        self.robot_data.qvel *= 0
        self.robot_data.ctrl *= 0
        self.robot_data.qacc *= 0

        if not booted:
            for i in range(1000):
                self.mj_step()
        else:
            for i in range(100):
                self.mj_step()

        self.robot_data.qvel *= 0
        self.robot_data.ctrl *= 0
        self.robot_data.qacc *= 0

        self.mj_step()

        self.reward = 0
        self.paws_on_ground = 0
        self.reward_components = np.zeros((1, 13))
        self.joint_not_in_range = False
        self.done = False
        # time.sleep(0.01)
        return

    def open_viewer(self, open_=True):
        if open_:
            self.view = mjv.launch_passive(self.robot, self.robot_data)
            self.view._shadows = False
            self.view.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True  # Enable contact points visualization
            self.view.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            
    def render(self):
        # viewer.launch(model=self.robot, data=self.robot_data)
        if self.first_render == 0:
            # self.set_camera()
            self.open_viewer()
            self.first_render = 1
            self.view.cam.trackbodyid = self.get_body_id('trunk')
        # self.camera.lookat = self.robot_data.body('trunk')
        self.view.sync()

    # def set_camera(self):
    #     self.camera = mj.MjvCamera()
    #     mj.mjv_defaultFreeCamera(self.robot, self.camera)
    #     self.camera.distance = 2
    #     self.camera.lookat = self.robot_data.body('trunk')

    def close(self):
        # ### TODO
        pass

if __name__ == "__main__":
    
    # xml_file = 'C:\\Users\\Omri\\Desktop\\MSc\\Thesis\\Simulation\\go1\\xml\\go1.xml'
    # env = Go1Env(xml=xml_file)
    # traj_gen = GenerateTrajectory([0, 0], [1, 15], 250)
    # lin = traj_gen.straight_line()
    # sin = traj_gen.sin_wave()
    # plt.figure(1)
    # plt.plot(lin[:, 0], lin[:, 1])
    # plt.figure(2)
    # plt.plot(sin[:, 0], sin[:, 1])
    # plt.show()
    pass
