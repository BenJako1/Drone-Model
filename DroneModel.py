import numpy as np
import matplotlib.pyplot as plt

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
            motor_vel = np.array([1000, 1000, 1000, 1000])
        elif 1.1 <= t < 1.2:
            motor_vel = np.array([1000, 1000, 1000, 1000])
        else:
           motor_vel = np.array([1000, 1000, 1000, 1000])
        
        return motor_vel
        
    def Simulate(self, dt, t_end):
        t = 0
        position_vec = np.zeros(3)
        velocity_vec = np.zeros(3)
        accel_vec = np.zeros(3)
        angle_vec = np.zeros(3)
        omega_vec = np.zeros(3)
        thrust_vec = np.zeros(3)
        torque_vec = np.zeros(3)
        
        self.position_store = self.position_vec
        self.velocity_store = self.velocity_vec
        self.angle_store = self.angle_vec
        self.torque_store = self.torque_vec
        self.time_store = [t]
        
        while t < t_end:
            
            k1_x = velocity_vec[0]
            k1_y = velocity_vec[1]
            k1_z = velocity_vec[2]
            
            k1_vx = accel_vec[0]
            k1_vy = accel_vec[1]
            k1_vz = accel_vec[2]
            
            k1_roll = angledot_vec[0]
            k1_pitch = angledot_vec[1]
            k1_yaw = angledot_vec[2]
            
            
            R_mat = self.frame_conversion_matrix(self.angle_vec[0], self.angle_vec[1], self.angle_vec[2])
            self.thrust_vec[2] = self.thrust_coefficient * (self.motor1_vel**2+self.motor2_vel**2+self.motor3_vel**2+self.motor4_vel**2)
            self.drag_vec = -self.drag * self.velocity_vec
            self.accel_vec = R_mat @ self.thrust_vec + np.array([0,0,-self.mass*self.gravity]) + self.drag_vec
            self.torque_vec = np.array([self.arm_length*self.thrust_coefficient*(self.motor1_vel**2-self.motor3_vel**2), self.arm_length*self.thrust_coefficient*(self.motor2_vel**2-self.motor4_vel**2), self.b_constant*(self.motor1_vel**2-self.motor2_vel**2+self.motor3_vel**2-self.motor4_vel**2)])
            self.omegadot_vec = np.linalg.inv(self.I) @ (self.torque_vec - np.cross(self.omega_vec, (self.I @ self.omega_vec)))
            
            self.omega_vec += self.omegadot_vec * dt
            
            self.angledot_vec = np.linalg.inv(self.omega_conversion_matrix(self.angle_vec[0], self.angle_vec[1], self.angle_vec[2])) @ self.omega_vec
            self.angle_vec += self.angledot_vec * dt
            
            self.velocity_vec += self.accel_vec * dt
            
            self.position_vec += self.velocity_vec * dt
            
            t += dt
            
            self.time_store.append(t)
            self.position = np.vstack([self.position_store, self.position_vec])
            self.angle = np.vstack([self.angle_store, self.angle_vec])
            self.torque = np.vstack([self.torque_store, self.torque_vec])
        
    def Display(self):
            fig, ax = plt.subplots(3, 2, figsize=(10,10))

            ax[0, 0].plot(self.time_store, self.angle[:,0], label='roll')
            ax[0, 0].set_title('roll angle')
            ax[0, 0].grid()

            ax[1, 0].plot(self.time_store, self.angle[:,1], label='pitch')
            ax[1, 0].set_title('pitch angle')
            ax[1, 0].grid()

            ax[2, 0].plot(self.time_store, self.angle[:,2], label='yaw')
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
test.Simulate(0.01, 10)
test.Display()

