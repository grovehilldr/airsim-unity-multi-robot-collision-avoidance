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


# Unused for now, will include later for speed.
# import quadprog as solver2





# Disable output of CVXOPT
options['show_progress'] = False
# Change default options of CVXOPT for faster solving
options['reltol'] = 1e-2 # was e-2
options['feastol'] = 1e-2 # was e-4
options['maxiters'] = 50 # default is 100


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
d1location = client.simGetObjectPose("Drone1")
d1x = d1location.position.x_val
d1y = d1location.position.y_val
d1z = d1location.position.z_val
print(d1x,d1y,d1z)

airsim.wait_key('Press any key to takeoff')
f1 = client.takeoffAsync(vehicle_name="Drone1")

f1.join()


state1 = client.getMultirotorState(vehicle_name="Drone1")
s = pprint.pformat(state1)
print("state: %s" % s)



d2x = -16.7
d2y = 25.29
d2z = -9

d3x = -16.7
d3y = -2.5
d3z = -9

d4x = -16.7
d4y = 20.26
d4z = -9

d5x = 11.3
d5y = -25.6
d5z = -9

d6x = 11.3
d6y = -2.5
d6z = -9

d7x = 11.3
d7y = 20.26
d7z = -9

d8x = 44.52
d8y = -25.6
d8z = -9

d9x = 44.52
d9y = -2.5
d9z = -9

d10x = 44.52
d10y = 20.26
d10z = -9


#setting starting points for Drone 1 and Drone 2
airsim.wait_key('Press any key to move vehicles')
f1 = client.moveToPositionAsync(44, 43, -5, 2, vehicle_name="Drone1")

#wait until they have moved
f1.join()
print("moved to initial position")
d1location = client.simGetObjectPose("Drone1")
d1x = d1location.position.x_val
d1y = d1location.position.y_val
d1z = d1location.position.z_val
print(d1x,d1y,d1z)

#Drone1's goal point
d1goal = [-27, -45 ,-5]


d1GToken = 0
d2GToken = 0
d3GToken = 0
d4GToken = 0

d1location = client.simGetObjectPose("Drone1")
d1x = d1location.position.x_val
d1y = d1location.position.y_val
d1z = d1location.position.z_val


d1gdistance = math.sqrt((d1x - d1goal[0])**2 + (d1y-d1goal[1])**2 + (d1z - d1goal[2])**2)







dxi = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
x = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])

def create_single_integrator_barrier_certificate(barrier_gain=.1, safety_radius=6, magnitude_limit=.5):


    def f(dxi, x):
   
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = 45 #4C2 
        A = np.zeros((num_constraints+1, 3*N))
        b = np.zeros(num_constraints+1)
        H = sparse(matrix(2*np.identity(3*N)))
        count = 0
        param = 1
        for i in range(N-1):
            for j in range(i+1, N):
                error = x[:, i] - x[:, j]
                h = (error[0]*error[0] + error[1]*error[1] +(error[2]/param)*(error[2]/param)) - np.power(safety_radius, 2)

                A[count, (2*i, (2*i+1), (2*i+2))] = -2*error
                A[count, (2*j+1, (2*j+2), (2*j+3))] = 2*error
                b[count] = barrier_gain*np.power(h, 3)

                count += 1

        #siwon added(
        zVec = 0 - d1z
        A[num_constraints] = [0,0,zVec,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        b[num_constraints] = 0
        #siwon added)
        # Threshold control inputs before QP

        norms = np.linalg.norm(dxi, 2, 0)
        
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] = dxi[:, idxs_to_normalize] * magnitude_limit/norms[idxs_to_normalize]
        #under this line, codes have been modified to fit in 3d positions

        f = -2*np.reshape(dxi, 3*N, order='F')
        #f = f.astype('float')

        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (3, -1), order='F')

    return f

#si_barrier_cert = create_single_integrator_barrier_certificate()
while(1):
    d1location = client.simGetObjectPose("Drone1")
    d1x = d1location.position.x_val
    d1y = d1location.position.y_val
    d1z = d1location.position.z_val

    # x = [1x, 2x, 3x, 4x]
    #     [1y, 2y, 3y, 4y]
    #     [1z, 2z, 3z, 4z]
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

    x[0][4] = d5x
    x[1][4] = d5y
    x[2][4] = d5z

    x[0][5] = d6x
    x[1][5] = d6y
    x[2][5] = d6z

    x[0][6] = d7x
    x[1][6] = d7y
    x[2][6] = d7z

    x[0][7] = d8x
    x[1][7] = d8y
    x[2][7] = d8z

    x[0][8] = d9x
    x[1][8] = d9y
    x[2][8] = d9z

    x[0][9] = d10x
    x[1][9] = d10y
    x[2][9] = d10z

    dxi[0][0] = d1goal[0] - x[0][0]
    dxi[1][0] = d1goal[1] - x[1][0]
    dxi[2][0] = d1goal[2] - x[2][0]

    dxi[0][1] = 0
    dxi[1][1] = 0
    dxi[2][1] = 0

    dxi[0][2] = 0
    dxi[1][2] = 0
    dxi[2][2] = 0

    dxi[0][3] = 0
    dxi[1][3] = 0
    dxi[2][3] = 0
    dxi[0][4] = 0
    dxi[1][4] = 0
    dxi[2][4] = 0

    dxi[0][5] = 0
    dxi[1][5] = 0
    dxi[2][5] = 0

    dxi[0][6] = 0
    dxi[1][6] = 0
    dxi[2][6] = 0

    dxi[0][7] = 0
    dxi[1][7] = 0
    dxi[2][7] = 0

    dxi[0][8] = 0
    dxi[1][8] = 0
    dxi[2][8] = 0

    dxi[0][9] = 0
    dxi[1][9] = 0
    dxi[2][9] = 0


    #dxi = dxi/2

    si_barrier_cert = create_single_integrator_barrier_certificate()
    dxi = si_barrier_cert(dxi,x)
    
    
#    client.moveToPositionAsync(dxi[0][0], dxi[1][0], dxi[2][0], 4, vehicle_name="Drone1")
#    client.moveToPositionAsync(dxi[0][1], dxi[1][1], dxi[2][1], 4, vehicle_name="Drone2")
    
    client.moveByVelocityAsync(dxi[0][0], dxi[1][0], dxi[2][0], 0.1, vehicle_name="Drone1")

    d1gdistance = math.sqrt((d1x - d1goal[0])**2 + (d1y-d1goal[1])**2 + (d1z - d1goal[2])**2)
