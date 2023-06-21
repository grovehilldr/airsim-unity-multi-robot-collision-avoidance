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

from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

from scipy.special import comb
#from rps.utilities.transformations import *

# Disable output of CVXOPT
options['show_progress'] = False
# Change default options of CVXOPT for faster solving
options['reltol'] = 1e-2 # was e-2
options['feastol'] = 1e-2 # was e-4
options['maxiters'] = 50 # default is 100
def trap_cdf_inv(a, c, delta, sigma):
    # returns list of b2, b1, sigma
    b2 = delta
    b1 = delta

    # a and c should be positive

    if a > c: # [-A, A] is the large one, and[-C, C] is the smaller one
        A = a
        C = c
    else:
        A = c
        C = a

    if A == 0 and C == 0:
        return b2, b1, sigma

    # O_vec = [-(A + C), -(A - C), (A - C), (A + C)] # vector of vertices on the trap distribution cdf

    h = 1 / (2 * A) # height of the trap distribution
    area_seq = [1/2 * 2 * C * h, 2 * (A - C) * h, 1/2 * 2 * C * h]
    area_vec = [area_seq[0], sum(area_seq[:2])]

    if abs(A - C) < 1e-5: # then is triangle
        # assuming sigma > 50
        b1 = (A + C) - 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1])) # 1 - area_vec[1] should be very close to 0.5
        b2 = -b1

        b1 = b1 + delta
        b2 = b2 + delta # apply shift here due to xi - xj

    else: # than is trap
        if sigma > area_vec[1]: # right triangle area
            b1 = (A + C) - 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))
            b2 = -(A + C) + 2 * C * np.sqrt((1 - sigma) / (1 - area_vec[1]))

            b1 = b1 + delta
            b2 = b2 + delta # apply shift here due to xi - xj

        elif sigma > area_vec[0] and sigma <= area_vec[1]: # in between the triangle part
            b1 = -(A - C) + (sigma - area_vec[0]) / h # assuming > 50%, then b1 should > 0
            b2 = -b1

            b1 = b1 + delta
            b2 = b2 + delta # apply shift here due to xi - xj

            # note that b1 could be > or < b2, depending on whether sigma > or < .5

        elif sigma <= area_vec[0]:
            b1 = -(A + C) + 2 * C * np.sqrt(sigma / area_vec[0]) # assuming > 50%, then b1 should > 0
            b2 = -b1

            b1 = b1 + delta
            b2 = b2 + delta # apply shift here due to xi - xj

        else:
            print('first triangle, which is not allowed as long as we assume sigma > 50%')

    return b2, b1, sigma
def create_si_pr_barrier_certificate_centralized(gamma = 1e4, safety_radius = 0.1, magnitude_limit = 0.2,Confidence = 1):
    assert isinstance(gamma, (int,
                              float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (gamma) must be an integer or float. Recieved type %r." % type(
        gamma).__name__
    assert isinstance(safety_radius, (int,
                                      float)), "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(
        safety_radius).__name__
    assert isinstance(magnitude_limit, (int,
                                        float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(
        magnitude_limit).__name__

    # Check user input ranges/sizes
    assert gamma > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (gamma) must be positive. Recieved %r." % gamma
    assert safety_radius >= 0.12, "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_radius
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit

    def barrier_certificate(dxi, x, XRandSpan, URandSpan):
        assert isinstance(dxi,
                          np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(
            dxi).__name__
        assert isinstance(x,
                          np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(
            x).__name__

        # Check user input ranges/sizes
        assert x.shape[
                   0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % \
                            x.shape[0]
        assert dxi.shape[
                   0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % \
                            dxi.shape[0]
        assert x.shape[1] == dxi.shape[
            1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (
            x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2))
        A = np.zeros((num_constraints, 2 * N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2 * np.identity(2 * N)))

        count = 0
        for i in range(N - 1):
            for j in range(i + 1, N):

                max_dvij_x = np.linalg.norm(URandSpan[0, i] + URandSpan[0, j])
                max_dvij_y = np.linalg.norm(URandSpan[1, i] + URandSpan[1, j])
                max_dxij_x = np.linalg.norm(x[0, i] - x[0, j]) + np.linalg.norm(XRandSpan[0, i] + XRandSpan[0, j])
                max_dxij_y = np.linalg.norm(x[1, i] - x[1, j]) + np.linalg.norm(XRandSpan[1, i] + XRandSpan[1, j])
                BB_x = -safety_radius ** 2 - 2 / gamma * max_dvij_x * max_dxij_x
                BB_y = -safety_radius ** 2 - 2 / gamma * max_dvij_y * max_dxij_y
                b2_x, b1_x, sigma = trap_cdf_inv(XRandSpan[0, i], XRandSpan[0, j], x[0, i] - x[0, j], Confidence)
                b2_y, b1_y, sigma = trap_cdf_inv(XRandSpan[1, i], XRandSpan[1, j], x[1, i] - x[1, j], Confidence)

                if (b2_x < 0 and b1_x > 0) or (b2_x > 0 and b1_x < 0):
                    # print('WARNING: distance between robots on x smaller than error bound!')
                    b_x = 0
                elif (b1_x < 0) and (b2_x < b1_x) or (b2_x < 0 and b2_x > b1_x):
                    b_x = b1_x
                elif (b2_x > 0 and b2_x < b1_x) or (b1_x > 0 and b2_x > b1_x):
                    b_x = b2_x
                else:
                    b_x = b1_x
                    # print('WARNING: no uncertainty or sigma = 0.5 on x')  # b1 = b2 or no uncertainty

                if (b2_y < 0 and b1_y > 0) or (b2_y > 0 and b1_y < 0):
                    # print('WARNING: distance between robots on y smaller than error bound!')
                    b_y = 0
                elif (b1_y < 0 and b2_y < b1_y) or (b2_y < 0 and b2_y > b1_y):
                    b_y = b1_y
                elif (b2_y > 0 and b2_y < b1_y) or (b1_y > 0 and b2_y > b1_y):
                    b_y = b2_y
                else:
                    b_y = b1_y

                A[count, (2 * i)] = -2 * b_x  # matlab original: A(count, (2*i-1):(2*i)) = -2*([b_x;b_y]);
                A[count, (2 * i + 1)] = -2 * b_y

                A[count, (2 * j)] = 2 * b_x  # matlab original: A(count, (2*j-1):(2*j)) =  2*([b_x;b_y])';
                A[count, (2 * j + 1)] = 2 * b_y

                h1 = np.linalg.norm([b_x, 0.0]) ** 2 - safety_radius ** 2 - 2 * np.linalg.norm(
                    [max_dvij_x, 0]) * np.linalg.norm([max_dxij_x, 0]) / gamma
                h2 = np.linalg.norm([0, b_y]) ** 2 - safety_radius ** 2 - 2 * np.linalg.norm(
                    [0, max_dvij_y]) * np.linalg.norm([0, max_dxij_y]) / gamma  # h_y

                h = h1 + h2

                b[count] = gamma * h ** 3  # matlab original: b(count) = gamma*h^3
                count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit / norms[idxs_to_normalize]

        f_mat = -2 * np.reshape(dxi, 2 * N, order='F')
        result = qp(H, matrix(f_mat), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')


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
N = dxi.shape[1]




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



v_rand_span = 0.005 * np.ones((2, N)) # setting up velocity error range for each robot

x_rand_span_x = 0.02 * np.random.randint(3, 4, (1, N)) # setting up position error range for each robot,
x_rand_span_y = 0.02 * np.random.randint(1, 4, (1, N)) # rand_span serves as the upper bound of uncertainty for each of the robot

x_rand_span_xy = np.concatenate((x_rand_span_x, x_rand_span_y))
#si_barrier_cert = create_single_integrator_barrier_certificate()
safety_radius = 0.20
confidence_level = 0.90
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
    #si_barrier_cert = create_single_integrator_barrier_certificate()
    si_barrier_cert = create_si_pr_barrier_certificate_centralized(safety_radius=safety_radius, Confidence=confidence_level)
    dxi[0][0] = dgoal[0][0] - x[0][0]
    dxi[1][0] = dgoal[1][0] - x[1][0]
    dxi[2][0] = dgoal[2][0] - x[2][0]
    dxi_r = si_barrier_cert(dxi,x,x_rand_span_xy,v_rand_span)
    
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

