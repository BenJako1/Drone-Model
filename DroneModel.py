import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

class Drone:
    def __init__(self, mass, I, L, k, drag, gravity=9.81, rho=1.293):
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
        
        self.motor1_vel = 0
        self.motor2_vel = 0
        self.motor3_vel = 0
        self.motor4_vel = 0
        
    def Initial_Conditions(self):
        self.position_vec = np.array([0, 0, 0])
        self.velocity_vec = np.array([0, 0, 0])
        self.angle_vec = np.array([0, 0, 0])
        self.omega_vec = np.array([0, 0, 0])
    
    def omega_conversion_matrix(self, phi, theta, psi):
        return np.array([[1, 0, np.sin(theta)],
                         [0, np.cos(phi), np.cos(theta)*np.sin(phi)],
                         [0, -np.sin(phi), np.cos(theta)*np.cos(phi)]])


    def frame_conversion_matrix(self, phi, theta, psi):
        return np.array([[np.cos(phi)*np.cos(psi)-np.cos(theta)*np.sin(phi)*np.sin(psi), -np.cos(psi)*np.sin(phi)-np.cos(phi)*np.cos(theta)*np.sin(psi), np.sin(theta)*np.sin(psi)],
                      [np.cos(theta)*np.cos(psi)*np.sin(phi)-np.cos(phi)*np.sin(psi), np.cos(phi)*np.cos(theta)*np.cos(psi)-np.sin(phi)*np.sin(psi), -np.cos(psi)*np.sin(theta)],
                      [np.sin(phi)*np.sin(theta), np.cos(phi)*np.sin(theta), np.cos(theta)]])
    
    def Control(self, t, y):
        pass
    
    def Equations_of_Motion(self, y):
        position_vec, velocity_vec, angle_vec, omega_vec = y
        
        R_mat = self.frame_conversion_matrix(angle_vec[0], angle_vec[1], angle_vec[2])
        thrust_vec = np.array([0, 0, self.thrust_coefficient * (self.motor1_vel**2+self.motor2_vel**2+self.motor3_vel**2+self.motor4_vel**2)])
        drag_vec = -self.drag * velocity_vec
        accel_vec = R_mat @ thrust_vec + np.array([0,0,-self.mass*self.gravity]) + drag_vec
        torque_vec = np.array([self.arm_length*self.thrust_coefficient*(self.motor1_vel**2-self.motor3_vel**2), self.arm_length*self.thrust_coefficient*(self.motor2_vel**2-self.motor4_vel**2), self.b_constant*(self.motor1_vel**2-self.motor2_vel**2+self.motor3_vel**2-self.motor4_vel**2)])
        
        omegadot_vec = np.linalg.inv(self.I) @ (torque_vec - np.cross(omega_vec, (self.I @ omega_vec)))
        angledot_vec = np.linalg.inv(self.omega_conversion_matrix(angle_vec[0], angle_vec[1], angle_vec[2])) @ omega_vec
        
        return velocity_vec, accel_vec, angledot_vec, omegadot_vec
    
    def Simulate(self, dt, t_end):
        
        self.data = integrate.solve_ivp(self.Control, [0, t_end], [self.position_vec, self.velocity_vec, self.angle_vec, self.omega_vec], t_eval=np.linspace(0, int(t_end), int(t_end * 50)))
        
    def Display(self):
            self.time_store = self.data.t
            self.angle_store = self.data[2]
            
            fig, ax = plt.subplots(3, 2, figsize=(10,10))

            ax[0, 0].plot(self.time_store, self.angle_store[:,0], label='roll')
            ax[0, 0].set_title('roll angle')
            ax[0, 0].grid()

            ax[1, 0].plot(self.time_store, self.angle_store[:,1], label='pitch')
            ax[1, 0].set_title('pitch angle')
            ax[1, 0].grid()

            ax[2, 0].plot(self.time_store, self.angle_store[:,2], label='yaw')
            ax[2, 0].set_title('yaw angle')
            ax[2, 0].grid()

            ax[0, 1].plot(self.time_store, self.torque_store[:,0], label='roll')
            ax[0, 1].set_title('roll torque')
            ax[0, 1].grid()

            ax[1, 1].plot(self.time_store, self.torque_store[:,1], label='pitch')
            ax[1, 1].set_title('pitch torque')
            ax[1, 1].grid()

            ax[2, 1].plot(self.time_store, self.torque_store[:,2], label='yaw')
            ax[2, 1].set_title('yaw torque')
            ax[2, 1].grid()

            plt.tight_layout()
            plt.show()
                
            
        
test = Drone(0.1, (0.01, 0.01, 0.1), 0.1, 0.000001, 0.6)
test.Initial_Conditions()
test.Simulate(0.01, 10)
test.Display()

