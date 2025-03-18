#Group members: Mahdis Moulai, Dina Amir Asadi, Shakial Hassanpour
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import config 

def euler_to_quat(psi, theta, phi):
    cy = np.cos(psi * 0.5)
    sy = np.sin(psi * 0.5)
    cp = np.cos(theta * 0.5)
    sp = np.sin(theta * 0.5)
    cr = np.cos(phi * 0.5)
    sr = np.sin(phi * 0.5)
    q0 = cy * cp * cr + sy * sp * sr
    q1 = cy * cp * sr - sy * sp * cr
    q2 = cy * sp * cr + sy * cp * sr
    q3 = sy * cp * cr - cy * sp * sr
    return np.array([q0, q1, q2, q3])

def quat_multiply(q, r):
    w1 = q[0]; x1 = q[1]; y1 = q[2]; z1 = q[3]
    w2 = r[0]; x2 = r[1]; y2 = r[2]; z2 = r[3]
    q0 = w1*w2 - x1*x2 - y1*y2 - z1*z2
    q1 = w1*x2 + x1*w2 + y1*z2 - z1*y2
    q2 = w1*y2 - x1*z2 + y1*w2 + z1*x2
    q3 = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([q0, q1, q2, q3])

def quat_to_euler(q):
    roll = np.arctan2(2*(q[:,0]*q[:,1] + q[:,2]*q[:,3]), 1 - 2*(q[:,1]**2 + q[:,2]**2))
    pitch = np.arcsin(2*(q[:,0]*q[:,2] - q[:,3]*q[:,1]))
    yaw = np.arctan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]), 1 - 2*(q[:,2]**2 + q[:,3]**2))
    return np.column_stack((roll, pitch, yaw))

def quat_to_rotation(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
    ])

def euler_to_rotation(phi, theta, psi):
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [ct*cy, ct*sy, -st],
        [sp*st*cy-cp*sy, sp*st*sy+cp*cy, sp*ct],
        [cp*st*cy+sp*sy, cp*st*sy-sp*cy, cp*ct]
    ])

def normalize_quat(q):
    return q / np.sqrt(np.sum(q**2))

def aircraft_dynamics_quat(t, state, forces_moments, Ixx, Iyy, Izz, Ixz, m, g, time):
    U, V, W = state[0:3]
    P, Q, R = state[3:6]
    q = state[6:10]
    X, Y, Z = state[10:13]
    
    idx = np.searchsorted(time, t, side='right') - 1
    idx = np.clip(idx, 0, len(time)-2)
    t1, t2 = time[idx], time[idx+1]
    f = (t - t1) / (t2 - t1)
    FM = (1-f) * forces_moments[idx] + f * forces_moments[idx+1]
    
    Fx, Fy, Fz, L, M, N = FM[:6] if len(FM) >= 6 else np.pad(FM, (0, 6-len(FM)), 'constant')
    
    Ud = (Fx/m) - g*np.sin(q[1]) + R*V - Q*W
    Vd = (Fy/m) + g*np.cos(q[1])*np.sin(q[0]) - R*U + P*W
    Wd = (Fz/m) + g*np.cos(q[1])*np.cos(q[0]) + Q*U - P*V
    
    I = np.array([[Ixx, 0, -Ixz], [0, Iyy, 0], [-Ixz, 0, Izz]])
    I_inv = np.linalg.inv(I)
    moments = np.array([L, M, N])
    cross_term = np.cross([P, Q, R], I @ [P, Q, R])
    ang_acc = I_inv @ (moments - cross_term)
    Pd, Qd, Rd = ang_acc
    
    omega = np.array([0, P, Q, R])
    dq_dt = 0.5 * quat_multiply(q, omega)
    q = normalize_quat(q)
    
    R_body_to_inertial = quat_to_rotation(q)
    vel_inertial = R_body_to_inertial @ [U, V, W]
    Xd, Yd, Zd = vel_inertial
    
    return np.concatenate(([Ud, Vd, Wd], [Pd, Qd, Rd], dq_dt, [Xd, Yd, Zd]))

def aircraft_dynamics_euler(t, state, forces_moments, Ixx, Iyy, Izz, Ixz, m, g, time):
    U, V, W = state[0:3]
    P, Q, R = state[3:6]
    phi, theta, psi = state[6:9]
    X, Y, Z = state[9:12]
    
    idx = np.searchsorted(time, t, side='right') - 1
    idx = np.clip(idx, 0, len(time)-2)
    t1, t2 = time[idx], time[idx+1]
    f = (t - t1) / (t2 - t1)
    FM = (1-f) * forces_moments[idx] + f * forces_moments[idx+1]
    
    Fx, Fy, Fz, L, M, N = FM[:6] if len(FM) >= 6 else np.pad(FM, (0, 6-len(FM)), 'constant')
    
    Ud = (Fx/m) - g*np.sin(theta) + R*V - Q*W
    Vd = (Fy/m) + g*np.cos(theta)*np.sin(phi) - R*U + P*W
    Wd = (Fz/m) + g*np.cos(theta)*np.cos(phi) + Q*U - P*V
    
    I = np.array([[Ixx, 0, -Ixz], [0, Iyy, 0], [-Ixz, 0, Izz]])
    I_inv = np.linalg.inv(I)
    moments = np.array([L, M, N])
    cross_term = np.cross([P, Q, R], I @ [P, Q, R])
    ang_acc = I_inv @ (moments - cross_term)
    Pd, Qd, Rd = ang_acc
    
    dphi_dt = P + (Q*np.sin(phi) + R*np.cos(phi))*np.tan(theta)
    dtheta_dt = Q*np.cos(phi) - R*np.sin(phi)
    dpsi_dt = (Q*np.sin(phi) + R*np.cos(phi))/np.cos(theta)
    
    R_body_to_inertial = euler_to_rotation(phi, theta, psi)
    vel_inertial = R_body_to_inertial @ [U, V, W]
    Xd, Yd, Zd = vel_inertial
    
    return np.concatenate(([Ud, Vd, Wd], [Pd, Qd, Rd], [dphi_dt, dtheta_dt, dpsi_dt], [Xd, Yd, Zd]))


try:
    data = np.loadtxt(config.DATA_FILE_PATH)
except FileNotFoundError:
    print(f"Error: data file not found at {config.DATA_FILE_PATH}!")
    exit(1)

time = data[:, 0]
forces_moments = data[:, 1:]

print(f"Data shape: {data.shape}")
print(f"Forces and moments shape: {forces_moments.shape}")
print(f"Sample forces_moments row: {forces_moments[0]}")

U0 = 264.6211; V0 = 0; W0 = 22.2674
P0 = 0; Q0 = 0; R0 = 0
phi0 = np.deg2rad(0); theta0 = np.deg2rad(1.81); psi0 = np.deg2rad(0)
X0 = 0; Y0 = 0; Z0 = 1000
q0 = euler_to_quat(psi0, theta0, phi0)

Ixx = 13.7e6; Iyy = 30.5e6; Izz = 43.1e6; Ixz = 0.83e6
m = 564000 / 32.2
g = 32.2

state0_quat = np.concatenate(([U0, V0, W0], [P0, Q0, R0], q0, [X0, Y0, Z0]))
state0_euler = np.concatenate(([U0, V0, W0], [P0, Q0, R0], [phi0, theta0, psi0], [X0, Y0, Z0]))

sol_quat = solve_ivp(lambda t, y: aircraft_dynamics_quat(t, y, forces_moments, Ixx, Iyy, Izz, Ixz, m, g, time),
                    [time[0], time[-1]], state0_quat, t_eval=time, method='RK45')
t_quat = sol_quat.t
state_quat = sol_quat.y.T

sol_euler = solve_ivp(lambda t, y: aircraft_dynamics_euler(t, y, forces_moments, Ixx, Iyy, Izz, Ixz, m, g, time),
                     [time[0], time[-1]], state0_euler, t_eval=time, method='RK45')
t_euler = sol_euler.t
state_euler = sol_euler.y.T

U_quat = state_quat[:, 0]; V_quat = state_quat[:, 1]; W_quat = state_quat[:, 2]
P_quat = state_quat[:, 3]; Q_quat = state_quat[:, 4]; R_quat = state_quat[:, 5]
quaternions = state_quat[:, 6:10]
X_quat = state_quat[:, 10]; Y_quat = state_quat[:, 11]; Z_quat = state_quat[:, 12]
eulerAngles_quat = quat_to_euler(quaternions)

U_euler = state_euler[:, 0]; V_euler = state_euler[:, 1]; W_euler = state_euler[:, 2]
P_euler = state_euler[:, 3]; Q_euler = state_euler[:, 4]; R_euler = state_euler[:, 5]
phi_euler = np.rad2deg(state_euler[:, 6]); theta_euler = np.rad2deg(state_euler[:, 7]); psi_euler = np.rad2deg(state_euler[:, 8])
X_euler = state_euler[:, 9]; Y_euler = state_euler[:, 10]; Z_euler = state_euler[:, 11]

vars_quat = [np.rad2deg(eulerAngles_quat[:, 0]), np.rad2deg(eulerAngles_quat[:, 1]), np.rad2deg(eulerAngles_quat[:, 2]),
             U_quat, V_quat, W_quat, P_quat, Q_quat, R_quat, X_quat, Y_quat, Z_quat, quaternions]
vars_euler = [phi_euler, theta_euler, psi_euler,
              U_euler, V_euler, W_euler, P_euler, Q_euler, R_euler, X_euler, Y_euler, Z_euler]

labels = ['Roll Angle (φ) [deg]', 'Pitch Angle (θ) [deg]', 'Yaw Angle (ψ) [deg]',
          'Forward Velocity (U) [ft/s]', 'Side Velocity (V) [ft/s]', 'Vertical Velocity (W) [ft/s]',
          'Roll Rate (P) [rad/s]', 'Pitch Rate (Q) [rad/s]', 'Yaw Rate (R) [rad/s]',
          'X Position [ft]', 'Y Position [ft]', 'Z Position [ft]', 'Quaternion Components']

for i in range(12):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_quat, vars_quat[i], 'b', label='Quaternion')
    plt.xlabel('Time (s)')
    plt.ylabel(labels[i])
    plt.title('Quaternion-Based ' + labels[i])
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(t_euler, vars_euler[i], 'r', label='Euler')
    plt.xlabel('Time (s)')
    plt.ylabel(labels[i])
    plt.title('Euler-Based ' + labels[i])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(t_quat, quaternions[:, i], 'b', label=f'q{i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel(f'q{i+1}')
    plt.title(f'Quaternion Component q{i+1}')
    plt.grid(True)
    plt.legend()
plt.tight_layout()

plt.show()

if __name__ == '__main__':
    pass
