from re import A
import airsim
import cv2
import numpy as np
import os
import pprint
import setup_path 
import tempfile
#siwon added
import math
import random

from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

from scipy.special import comb

import time


# Unused for now, will include later for speed.
# import quadprog as solver2





# Disable output of CVXOPT
options['show_progress'] = False
# Change default options of CVXOPT for faster solving
options['reltol'] = 1e-5 # was e-2
options['feastol'] = 1e-5 # was e-4
options['maxiters'] = 150 # default is 100


# Use below in settings.json with Blocks environment
"""
{
	"SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
	"SettingsVersion": 1.2,
	"SimMode": "Multirotor",
	"ClockSpeed": 1,
	
	"Vehicles": {
		"Drone1": {
		  "VehicleType": "SimpleFlight",
		  "X": -5, "Y": 0, "Z": -2
		},
		"Drone2": {
		  "VehicleType": "SimpleFlight",
		  "X": 10, "Y": 0, "Z": -2
		}

    }
}
"""

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, "Drone1")


client.armDisarm(True, "Drone1")



airsim.wait_key('Press any key to takeoff')
f1 = client.takeoffAsync(vehicle_name="Drone1")


f1.join()


state1 = client.getMultirotorState(vehicle_name="Drone1")
s = pprint.pformat(state1)



d1location = client.simGetObjectPose("Drone1")


print(d1location)

print("hlihlihlihlh")
#setting starting points for Drone 1 and Drone 2

airsim.wait_key('Press any key to move vehicles')
f1 = client.moveToPositionAsync(0, -20, -7, 2, vehicle_name="Drone1")


#wait until they have moved
f1.join()




#dgoal = np.array([ [10,10,0,0], [7,-10,-10,7] , [-1,-1,-1,-1] ])

d1GToken = 0


d1location = client.simGetObjectPose("Drone1")
d1x = d1location.position.x_val
d1y = d1location.position.y_val
d1z = d1location.position.z_val

dxi = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])

x = np.array([[0,0,0,0],[0,0,0,0],[-2,-4,-6,-8]])





dgoal = np.array([ [0,1,1,1], [25,1,1,1] , [-5,1,1,1] ])


def create_single_integrator_barrier_certificate(barrier_gain=30, safety_radius=1.5, magnitude_limit=.5,boundary_points = np.array([-30,30,-30,30,-8.6,0])):


    def f(dxi, x):
   
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = 3+2
        A = np.zeros((num_constraints, 3*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2*np.identity(3*N)))

        count = 0
        for i in range(1):      #drone num #fix when changed
            for j in range(i+1, N):
                error = x[:, i] - x[:, j]
                h = (error[0]*error[0] + error[1]*error[1] +error[2]*error[2]) - np.power(safety_radius, 2) #fix when changed

                A[count, (3*i, (3*i+1), (3*i+2))] = -2*error
                #A[count, (3*j, (3*j+1), (3*j+2))] = 2*error
                b[count] = barrier_gain*np.power(h, 3)

                count += 1
        

        for k in range(1):
            

            #Pos z
            #tmp = k-1
            A[count, (3*k,3*k+1, 3*k+2)] = np.array([0,0,1])
            b[count] = 0.4*barrier_gain*(boundary_points[5] - safety_radius/2 - x[2,k])**3;
            count += 1

            #Neg z
            A[count, (3*k,3*k+1, 3*k+2)] = -np.array([0,0,1])
            b[count] = 0.4*barrier_gain*(-boundary_points[4] - safety_radius/2 + x[2,k])**3;
            count += 1


        norms = np.linalg.norm(dxi, 2, 0)
        
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] = dxi[:, idxs_to_normalize] * magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, 3*N, order='F')
        f = f.astype('float')

        result = qp(H, matrix(f), matrix(A), matrix(b))['x']
        


        return np.reshape(result, (3, -1), order='F')

    return f
kfcount = 0
dt = 0
A = np.array([[1,0,0],[0,1,0],[0,0,1]])

H = np.array([[1,0,0],[0,1,0],[0,0,1]])
Q = np.array([[0.01,0,0],[0,0.01,0],[0,0,0.01]])
R = np.array([[0.01,0,0],[0,0.01,0],[0,0,0.01]])
u_esti =  np.array([ [dxi[0][0]],[dxi[1][0]],[dxi[2][0]] ])

rx = np.array([[0],[0],[0]])

if(kfcount == 0):
    P = np.array([[0.1, 0, 0],[0, 0.1,0],[0, 0, 0.1]])
    x_esti = np.array([ [x[0][0]],[x[1][0]],[x[2][0]] ])
    z_meas = np.array([ ([rx[0][0]]),([rx[0][0]]),([rx[0][0]]) ])
def kalman_filter(z_meas, x_esti, P):
    """Kalman Filter Algorithm."""
    # (1) Prediction.
    
    x_pred = A @ x_esti + dt*u_esti
    P_pred = A @ P @ A.T + Q
 
    # (2) Kalman Gain.
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
 
    # (3) Estimation.
    x_esti = x_pred + K @ (z_meas - H @ x_pred)
 
    # (4) Error Covariance.
    P = P_pred - K @ H @ P_pred
 
    return x_esti, P




#si_barrier_cert = create_single_integrator_barrier_certificate()
while(1):
    start_time = time.time()
    d1location = client.simGetObjectPose("Drone1")
    d1x = d1location.position.x_val
    d1y = d1location.position.y_val
    d1z = d1location.position.z_val
    

    x[0][0] = d1x
    x[1][0] = d1y
    x[2][0] = d1z
    rx[0][0] = x[0][0] +0.015
    rx[1][0] = x[1][0]+0.015
    rx[2][0] = x[2][0]-0.015
    z_meas = np.array([ ([rx[0][0]]),([rx[0][0]]),([rx[0][0]]) ])


    dxi[0][0] = dgoal[0][0] - x[0][0]
    dxi[1][0] = dgoal[1][0] - x[1][0]
    dxi[2][0] = dgoal[2][0] - x[2][0]
    dxi/5
    si_barrier_cert = create_single_integrator_barrier_certificate()
    dxi = si_barrier_cert(dxi,x)

    
#    client.moveToPositionAsync(dxi[0][0], dxi[1][0], dxi[2][0], 4, vehicle_name="Drone1")
#    client.moveToPositionAsync(dxi[0][1], dxi[1][1], dxi[2][1], 4, vehicle_name="Drone2")
    
    client.moveByVelocityAsync(dxi[0][0], dxi[1][0], dxi[2][0], 1, vehicle_name="Drone1")
    end_time = time.time()
    dt = end_time - start_time
    #client.moveByVelocityAsync(dxi[0][1], dxi[1][1], dxi[2][1], 0.1, vehicle_name="Drone2")
    #client.moveByVelocityAsync(dxi[0][2], dxi[1][2], dxi[2][2], 0.1, vehicle_name="Drone3")
    #client.moveByVelocityAsync(dxi[0][3], dxi[1][3], dxi[2][3], 0.1, vehicle_name="Drone4")

    kalman_filter(z_meas, x_esti, P)
    print("x error", d1x - x_esti[0])

    d1gdistance = math.sqrt((d1x - dgoal[0][0])**2 + (d1y-dgoal[1][0])**2 + (d1z - dgoal[2][0])**2)

    if (d1gdistance < 1.5):
        if (d1GToken != 1):
            print("Drone 1 reached very close to the goal")

        d1GToken = 1

  




        
        
    



        
        
        
        
    #else:
     #   client.moveToPositionAsync( (d1goal[0]), (d1goal[1]), (d1goal[2]), 3, vehicle_name="Drone1")
      #  print("step")









airsim.wait_key('Press any key to reset to original state')

client.armDisarm(False, "Drone1")


client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False, "Drone1")

