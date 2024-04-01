import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

class Drone:
    def __init__(self, mass, I, L, k, drag, gravity=9.81, rho=1.293, disturbance=10):
        self.mass = mass
        self.I = np.array([[I[0], 0, 0],
                          [0, I[1], 0],
                          [0, 0, I[2]]])
        self.arm_length = L
        self.thrust_coefficient = k
        self.drag_coefficient = drag
        self.b_constant = 0.5 * 0.05**3 * 0.2 * 0.5 * 0.005
        self.drag = drag
        
        self.gravity = gravity
        self.rho = rho
        self.disturbance = disturbance
    
    def omega_conversion_matrix(self, phi, theta, psi):
        return np.array([[1, 0, np.sin(theta)],
                         [0, np.cos(phi), np.cos(theta)*np.sin(phi)],
                         [0, -np.sin(phi), np.cos(theta)*np.cos(phi)]])


    def frame_conversion_matrix(self, phi, theta, psi):
        return np.array([[np.cos(phi)*np.cos(psi)-np.cos(theta)*np.sin(phi)*np.sin(psi), -np.cos(psi)*np.sin(phi)-np.cos(phi)*np.cos(theta)*np.sin(psi), np.sin(theta)*np.sin(psi)],
                      [np.cos(theta)*np.cos(psi)*np.sin(phi)-np.cos(phi)*np.sin(psi), np.cos(phi)*np.cos(theta)*np.cos(psi)-np.sin(phi)*np.sin(psi), -np.cos(psi)*np.sin(theta)],
                      [np.sin(phi)*np.sin(theta), np.cos(phi)*np.sin(theta), np.cos(theta)]])
    
    def Control(self, t, y):
        if 1 < t < 1.1:
            motor_vel = np.array([1198.5, 1198.5, 1300, 1300])
        elif 1.1 < t < 1.2:
            motor_vel = np.array([1300, 1300, 1204, 1204])
        else:
            motor_vel = np.array([1250, 1250, 1250, 1250])

        return self.Equations_of_Motion(y, motor_vel)
    
    def Equations_of_Motion(self, y, motor_vel):
        x, y, z, vx, vy, vz, roll, pitch, yaw, omega1, omega2, omega3 = y
        
        velocity_vec = np.array([vx, vy, vz])
        angle_vec = np.array([roll, pitch, yaw])
        omega_vec = np.array([omega1, omega2, omega3])
        
        torque_vec = np.array([self.arm_length/np.sqrt(2)*self.thrust_coefficient*(motor_vel[3]**2-motor_vel[2]**2-motor_vel[1]**2+motor_vel[0]**2), self.arm_length/np.sqrt(2)*self.thrust_coefficient*(motor_vel[3]**2+motor_vel[2]**2-motor_vel[1]**2-motor_vel[0]**2), self.b_constant*(motor_vel[0]**2-motor_vel[1]**2+motor_vel[2]**2-motor_vel[3]**2)])
        R_mat = self.frame_conversion_matrix(angle_vec[0], angle_vec[1], angle_vec[2])
        thrust_vec = np.array([0, 0, self.thrust_coefficient * (motor_vel[0]**2+motor_vel[1]**2+motor_vel[2]**2+motor_vel[3]**2)])
        drag_vec = -self.drag * velocity_vec
        
        accel_vec = R_mat @ thrust_vec / self.mass + np.array([0,0,-self.gravity]) + drag_vec/self.mass
        omegadot_vec = np.linalg.inv(self.I) @ (torque_vec - np.cross(omega_vec, (self.I @ omega_vec)))
        angledot_vec = np.linalg.inv(self.omega_conversion_matrix(angle_vec[0], angle_vec[1], angle_vec[2])) @ omega_vec
        
        return vx, vy, vz, accel_vec[0], accel_vec[1], accel_vec[2], angledot_vec[0], angledot_vec[1], angledot_vec[2], omegadot_vec[0], omegadot_vec[1], omegadot_vec[2]
    
    def Simulate(self, t_end):
        
        self.data = integrate.solve_ivp(self.Control, [0, t_end], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], t_eval=np.linspace(0, int(t_end), int(t_end * 50)), max_step=0.02)
        
    def Display(self):
        self.time_store = self.data.t
        self.position_store = np.array([self.data.y[0], self.data.y[1], self.data.y[2]])
        self.angle_store = np.array([self.data.y[6], self.data.y[7], self.data.y[8]])
        
        fig, ax = plt.subplots(3, 2, figsize=(10,10))

        ax[0, 0].plot(self.time_store, self.angle_store[0], label='roll')
        ax[0, 0].set_title('roll angle')
        ax[0, 0].grid()

        ax[1, 0].plot(self.time_store, self.angle_store[1], label='pitch')
        ax[1, 0].set_title('pitch angle')
        ax[1, 0].grid()

        ax[2, 0].plot(self.time_store, self.angle_store[2], label='yaw')
        ax[2, 0].set_title('yaw angle')
        ax[2, 0].grid()

        ax[0, 1].plot(self.time_store, self.position_store[0], label='roll')
        ax[0, 1].set_title('X')
        ax[0, 1].grid()

        ax[1, 1].plot(self.time_store, self.position_store[1], label='pitch')
        ax[1, 1].set_title('Y')
        ax[1, 1].grid()

        ax[2, 1].plot(self.time_store, self.position_store[2], label='yaw')
        ax[2, 1].set_title('Z')
        ax[2, 1].grid()

        plt.tight_layout()
        plt.show()
                
            

test = Drone(0.1, (0.01, 0.01, 0.1), 0.1, 0.000001, 0.6)
test.Simulate(2)
test.Display()

