from math import radians
import numpy as np
import matplotlib.pyplot as plt
import random


# Testing data
def DrawSlotPts(originX, originY, length, radius, rotationDeg):
    thetaDeg = np.linspace(0, 360, 41)
    thetaRad = 2* np.pi / 360 * thetaDeg

    coorX = [radius*(1 + random.random()/30) * np.cos(rad) for rad in thetaRad]
    coorY = [radius*(1 + random.random()/30) * np.sin(rad) for rad in thetaRad]

    coorX = [coor-length/2 if coor<0 else coor+length/2 for coor in coorX]

    rotationRad = 2 * np.pi / 360 *rotationDeg
    for itr in range(len(coorX)):
        xItr = coorX[itr]
        yItr = coorY[itr]

        radiusItr = (xItr**2 + yItr**2)**0.5
        artTan = np.arctan(abs(yItr)/abs(xItr))

        if xItr >= 0 and yItr >= 0:
            thetaItr = artTan
        elif xItr < 0 and yItr >= 0:
            thetaItr = np.pi - artTan
        elif xItr < 0 and yItr < 0:
            thetaItr = np.pi + artTan
        else:
            thetaItr = 2 * np.pi - artTan

        coorX[itr] = radiusItr * np.cos(rotationRad+thetaItr) + originX
        coorY[itr] = radiusItr * np.sin(rotationRad+thetaItr) + originY
        
    Plot(coorX, coorY, originX, originY, radius)
    
    return coorX, coorY


def Plot(coorX, coorY, originX, originY, radius):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(coorX, coorY, '+')
    ax.set_xlim([originX-2*radius, originX+2*radius])
    ax.set_ylim([originY-2*radius, originY+2*radius])
    ax.grid()
    plt.show()

    return fig, ax

# Hypothesis model



# Main
if __name__ == '__main__':
    originX     = 1000
    originY     = 800
    length      = 80
    radius      = 100
    rotationDeg = 30

    coorX, coorY = DrawSlotPts(originX=originX, originY=originY, length=length, radius=radius, rotationDeg=rotationDeg)
    answer = coorX, coorY

    initial = np.array([0, 0, 50, 50, 0]).reshape((5,1))