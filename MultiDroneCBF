# This file controls 4 drones and make them swap positions and move back to their original location.
# Modified Safety barrier certificate function originally provided by Georgia Tech.
from re import A
import airsim
import cv2
import numpy as np
import os
import pprint
import setup_path 
import tempfile
import math
import random
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

# Disable output of CVXOPT
options['show_progress'] = False
# Change default options of CVXOPT for faster solving
options['reltol'] = 1e-2 # was e-2
options['feastol'] = 1e-2 # was e-4
options['maxiters'] = 50 # default is 100

# Copy and paste this to your settings.json file in \Documents\AirSim
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
		  "X": 0, "Y": 0, "Z": -2
		},
		"Drone3": {
		  "VehicleType": "SimpleFlight",
		  "X": 5, "Y": 0, "Z": -2
		},
		"Drone4": {
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
client.enableApiControl(True, "Drone2")
client.enableApiControl(True, "Drone3")
client.enableApiControl(True, "Drone4")
client.armDisarm(True, "Drone1")
client.armDisarm(True, "Drone2")
client.armDisarm(True, "Drone3")
client.armDisarm(True, "Drone4")

airsim.wait_key('Press any key to takeoff')
f1 = client.takeoffAsync(vehicle_name="Drone1")
f2 = client.takeoffAsync(vehicle_name="Drone2")
f3 = client.takeoffAsync(vehicle_name="Drone3")
f4 = client.takeoffAsync(vehicle_name="Drone4")

# wait until the drone have taken off
f1.join()
f2.join()
f3.join()
f4.join()

state1 = client.getMultirotorState(vehicle_name="Drone1")
s = pprint.pformat(state1)
print("state: %s" % s)
state2 = client.getMultirotorState(vehicle_name="Drone2")
s = pprint.pformat(state2)
print("state: %s" % s)
state3 = client.getMultirotorState(vehicle_name="Drone3")
s = pprint.pformat(state3)
print("state: %s" % s)
state4 = client.getMultirotorState(vehicle_name="Drone4")
s = pprint.pformat(state4)
print("state: %s" % s)

airsim.wait_key('Press any key to move vehicles')
f1 = client.moveToPositionAsync(0, -10, -8, 5, vehicle_name="Drone1")
f2 = client.moveToPositionAsync(0, 7, -8, 5, vehicle_name="Drone2")
f3 = client.moveToPositionAsync(10, 7, -8, 5, vehicle_name="Drone3")
f4 = client.moveToPositionAsync(10, -10, -8, 5, vehicle_name="Drone4")

# wait until the drone have moved to their starting points
f1.join()
f2.join()
f3.join()
f4.join()

d1goal = [10,7,-8]
d2goal = [10,-10,-8]
d3goal = [0,-10,-8]
d4goal = [0,7,-8]

d1GToken = 0
d2GToken = 0
d3GToken = 0
d4GToken = 0

dxi = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
x = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])

def create_single_integrator_barrier_certificate(barrier_gain=.1, safety_radius=3, magnitude_limit=.5):


    def f(dxi, x):
   
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = 6 #4C2 
        A = np.zeros((num_constraints, 3*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2*np.identity(3*N)))

        count = 0
        for i in range(N-1):
            for j in range(i+1, N):
                error = x[:, i] - x[:, j]
                h = (error[0]*error[0] + error[1]*error[1] + error[2]*error[2]) - np.power(safety_radius, 2)

                A[count, (2*i, (2*i+1), (2*i+2))] = -2*error
                A[count, (2*j+1, (2*j+2), (2*j+3))] = 2*error
                b[count] = barrier_gain*np.power(h, 3)

                count += 1

        # Threshold control inputs before QP

        norms = np.linalg.norm(dxi, 2, 0)
        
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] = dxi[:, idxs_to_normalize] * magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, 3*N, order='F')

        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (3, -1), order='F')

    return f

while(1):
    d1location = client.simGetObjectPose("Drone1")
    d1x = d1location.position.x_val
    d1y = d1location.position.y_val
    d1z = d1location.position.z_val
    d2location = client.simGetObjectPose("Drone2")
    d2x = d2location.position.x_val
    d2y = d2location.position.y_val
    d2z = d2location.position.z_val
    d3location = client.simGetObjectPose("Drone3")
    d3x = d3location.position.x_val
    d3y = d3location.position.y_val
    d3z = d3location.position.z_val
    d4location = client.simGetObjectPose("Drone4")
    d4x = d4location.position.x_val
    d4y = d4location.position.y_val
    d4z = d4location.position.z_val
	
    # x = [d1x, d2x, d3x, d4x]
    #     [d1y, d2y, d3y, d4y]
    #     [d1z, d2z, d3z, d4z]
    x[0][0] = d1x
    x[1][0] = d1y
    x[2][0] = d1z

    x[0][1] = d2x
    x[1][1] = d2y
    x[2][1] = d2z

    x[0][2] = d3x
    x[1][2] = d3y
    x[2][2] = d3z

    x[0][3] = d4x
    x[1][3] = d4y
    x[2][3] = d4z

    dxi[0][0] = d1goal[0] - x[0][0]
    dxi[1][0] = d1goal[1] - x[1][0]
    dxi[2][0] = d1goal[2] - x[2][0]

    dxi[0][1] = d2goal[0] - x[0][1]
    dxi[1][1] = d2goal[1] - x[1][1]
    dxi[2][1] = d2goal[2] - x[2][1]

    dxi[0][2] = d3goal[0] - x[0][2]
    dxi[1][2] = d3goal[1] - x[1][2]
    dxi[2][2] = d3goal[2] - x[2][2]

    dxi[0][3] = d4goal[0] - x[0][3]
    dxi[1][3] = d4goal[1] - x[1][3]
    dxi[2][3] = d4goal[2] - x[2][3]

    dxi = dxi/5

    si_barrier_cert = create_single_integrator_barrier_certificate()
    dxi = si_barrier_cert(dxi,x)
    
    client.moveByVelocityAsync(dxi[0][0], dxi[1][0], dxi[2][0], 0.1, vehicle_name="Drone1")
    client.moveByVelocityAsync(dxi[0][1], dxi[1][1], dxi[2][1], 0.1, vehicle_name="Drone2")
    client.moveByVelocityAsync(dxi[0][2], dxi[1][2], dxi[2][2], 0.1, vehicle_name="Drone3")
    client.moveByVelocityAsync(dxi[0][3], dxi[1][3], dxi[2][3], 0.1, vehicle_name="Drone4")
    d1gdistance = math.sqrt((d1x - d1goal[0])**2 + (d1y-d1goal[1])**2 + (d1z - d1goal[2])**2)
    d2gdistance = math.sqrt((d2x - d2goal[0])**2 + (d2y-d2goal[1])**2 + (d2z - d2goal[2])**2)
    d3gdistance = math.sqrt((d3x - d3goal[0])**2 + (d3y-d3goal[1])**2 + (d3z - d3goal[2])**2)
    d4gdistance = math.sqrt((d4x - d4goal[0])**2 + (d4y-d4goal[1])**2 + (d4z - d4goal[2])**2)
    
    if (d1gdistance < 1.5):
        if (d1GToken != 1):
            print("Drone 1 reached very close to the goal")

        d1GToken = 1

    if (d2gdistance < 1.5):
        if (d2GToken != 1):
            print("Drone 2 reached very close to the goal")
        d2GToken = 1
        
    if (d3gdistance < 1.5):
        if (d3GToken != 1):
            print("Drone 3 reached very close to the goal")
        d3GToken = 1
   
    if (d4gdistance <1.5):
        if (d4GToken != 1):
            print("Drone 4 reached very close to the goal")
        d4GToken = 1
            
    if(d1GToken == 1 and d2GToken == 1 and d3GToken == 1 and d4GToken == 1):
        print("all drones are at the goal, now going back to starting point")
        break

d1goal = [0,-10,-8]
d2goal = [0,7,-8]
d3goal = [10,7,-8]
d4goal = [10,-10,-8]

while(1):
    d1location = client.simGetObjectPose("Drone1")
    d1x = d1location.position.x_val
    d1y = d1location.position.y_val
    d1z = d1location.position.z_val
    d2location = client.simGetObjectPose("Drone2")
    d2x = d2location.position.x_val
    d2y = d2location.position.y_val
    d2z = d2location.position.z_val
    d3location = client.simGetObjectPose("Drone3")
    d3x = d3location.position.x_val
    d3y = d3location.position.y_val
    d3z = d3location.position.z_val
    d4location = client.simGetObjectPose("Drone4")
    d4x = d4location.position.x_val
    d4y = d4location.position.y_val
    d4z = d4location.position.z_val
    # x = [d1x, d2x, d3x, d4x]
    #     [d1y, d2y, d3y, d4y]
    #     [d1z, d2z, d3z, d4z]
    x[0][0] = d1x
    x[1][0] = d1y
    x[2][0] = d1z

    x[0][1] = d2x
    x[1][1] = d2y
    x[2][1] = d2z

    x[0][2] = d3x
    x[1][2] = d3y
    x[2][2] = d3z

    x[0][3] = d4x
    x[1][3] = d4y
    x[2][3] = d4z

    dxi[0][0] = d1goal[0] - x[0][0]
    dxi[1][0] = d1goal[1] - x[1][0]
    dxi[2][0] = d1goal[2] - x[2][0]

    dxi[0][1] = d2goal[0] - x[0][1]
    dxi[1][1] = d2goal[1] - x[1][1]
    dxi[2][1] = d2goal[2] - x[2][1]

    dxi[0][2] = d3goal[0] - x[0][2]
    dxi[1][2] = d3goal[1] - x[1][2]
    dxi[2][2] = d3goal[2] - x[2][2]

    dxi[0][3] = d4goal[0] - x[0][3]
    dxi[1][3] = d4goal[1] - x[1][3]
    dxi[2][3] = d4goal[2] - x[2][3]

    #dxi = dxi/10 ->original
    dxi = dxi /5
    si_barrier_cert = create_single_integrator_barrier_certificate()
    dxi = si_barrier_cert(dxi,x)

    client.moveByVelocityAsync(dxi[0][0], dxi[1][0], dxi[2][0], 0.1, vehicle_name="Drone1")
    client.moveByVelocityAsync(dxi[0][1], dxi[1][1], dxi[2][1], 0.1, vehicle_name="Drone2")
    client.moveByVelocityAsync(dxi[0][2], dxi[1][2], dxi[2][2], 0.1, vehicle_name="Drone3")
    client.moveByVelocityAsync(dxi[0][3], dxi[1][3], dxi[2][3], 0.1, vehicle_name="Drone4")
    d1gdistance = math.sqrt((d1x - d1goal[0])**2 + (d1y-d1goal[1])**2 + (d1z - d1goal[2])**2)
    d2gdistance = math.sqrt((d2x - d2goal[0])**2 + (d2y-d2goal[1])**2 + (d2z - d2goal[2])**2)
    d3gdistance = math.sqrt((d3x - d3goal[0])**2 + (d3y-d3goal[1])**2 + (d3z - d3goal[2])**2)
    d4gdistance = math.sqrt((d4x - d4goal[0])**2 + (d4y-d4goal[1])**2 + (d4z - d4goal[2])**2)
    if (d1gdistance < 1.5):
        if (d1GToken != 2):
            print("Drone 1 reached very close to the goal")
        d1GToken = 2

    if (d2gdistance < 1.5):
        if (d2GToken != 2):
            print("Drone 2 reached very close to the goal")

        d2GToken = 2
        
    if (d3gdistance < 1.5):
        if (d3GToken != 2):
            print("Drone 3 reached very close to the goal")

        d3GToken = 2
   
    if (d4gdistance <1.5):
        if (d4GToken != 2):
            print("Drone 4 reached very close to the goal")
 
        d4GToken = 2
            
    if(d1GToken == 2 and d2GToken == 2 and d3GToken == 2 and d4GToken == 2):
        print("all drones are at the goal")
        break            

airsim.wait_key('Press any key to reset to original state')

client.armDisarm(False, "Drone1")
client.armDisarm(False, "Drone2")
client.armDisarm(False, "Drone3")
client.armDisarm(False, "Drone4")
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False, "Drone1")
client.enableApiControl(False, "Drone2")
client.enableApiControl(False, "Drone3")
client.enableApiControl(False, "Drone4")
