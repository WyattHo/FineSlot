import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.optimize


# Testing data
def DrawSlotPts(transX, transY, length, radius, rotationDeg):
    angleDeg = np.linspace(0, 360, 31)
    angleRad = 2* np.pi / 360 * angleDeg

    coorX = [radius*(1 + random.random()/60) * np.cos(rad) for rad in angleRad]
    coorY = [radius*(1 + random.random()/60) * np.sin(rad) for rad in angleRad]

    coorX = [coor-length/2 if coor<0 else coor+length/2 for coor in coorX]

    coorXY = []
    rotationRad = 2 * np.pi / 360 * rotationDeg
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

        coorX[itr] = radiusItr * np.cos(rotationRad+thetaItr) + transX
        coorY[itr] = radiusItr * np.sin(rotationRad+thetaItr) + transY

        coorXY.append((coorX[itr], coorY[itr]))
    
    return tuple(coorXY)


def myPlot(*arguments, coorXY):
    coorX = [coor[0] for coor in coorXY]
    coorY = [coor[1] for coor in coorXY]

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(coorX, coorY, '+', markersize=10, markeredgewidth=2, markerfacecolor='royalblue', markeredgecolor='royalblue')

    for coor in arguments:
        ax.plot(coor[0], coor[1], '+', markersize=10, markeredgewidth=2, markerfacecolor='orange', markeredgecolor='orange')

    ax.grid()
    plt.show()

    return fig, ax

# Hypothesis model
def FindTranslations(coorXY):
    maxX = max([coor[0] for coor in coorXY])
    minX = min([coor[0] for coor in coorXY])
    
    maxY = max([coor[1] for coor in coorXY])
    minY = min([coor[1] for coor in coorXY])

    return ((maxX+minX)/2, (maxY+minY)/2)


def FindLengthRadius(coorXY, trans, angleRad, rotationRad):
    distance = [((coor[0]-trans[0])**2 + (coor[1]-trans[1])**2)**0.5 for coor in coorXY]
    maxDist = max(distance)
    yProject = [dist * abs(np.sin(angleRad[itr]-rotationRad))  for itr, dist in enumerate(distance)]

    radius = max(yProject)
    length = 2 * (maxDist - radius)

    return length, radius


def FindSlotRotation(coorXY, trans):
    distance = [((coor[0]-trans[0])**2 + (coor[1]-trans[1])**2)**0.5 for coor in coorXY]
    
    maxDist = 0
    for itr, dist in enumerate(distance):
        if dist > maxDist and (coorXY[itr][0]-trans[0]) > 0:
            maxDist = dist
    
    maxId = distance.index(maxDist)
    
    deltaX = coorXY[maxId][0]-trans[0]
    deltaY = coorXY[maxId][1]-trans[1]

    if deltaY > 0:
        rotationRad = np.arctan(abs(deltaY)/abs(deltaX))
    else:
        rotationRad = 2 * np.pi - np.arctan(abs(deltaY)/abs(deltaX))

    return rotationRad


def FindPtsAngle(coorXY, trans, rotationDeg):
    rotationRad = 2 * np.pi / 360 *rotationDeg
    angleRad = []
    for coor in coorXY:
        deltaX = coor[0] - trans[0]
        deltaY = coor[1] - trans[1]

        if deltaX >=0 and deltaY >= 0:
            theta = np.arctan(abs(deltaY)/abs(deltaX)) + rotationRad
        elif deltaX < 0 and deltaY >=0:
            theta = np.pi - np.arctan(abs(deltaY)/abs(deltaX)) + rotationRad
        elif deltaX < 0 and deltaY < 0:
            theta = np.pi + np.arctan(abs(deltaY)/abs(deltaX)) + rotationRad
        else:
            theta = 2 * np.pi - np.arctan(abs(deltaY)/abs(deltaX)) + rotationRad

        if theta > 2 * np.pi:
            theta -= 2 * np.pi

        angleRad.append([theta])
    
    return angleRad

        
def GetCost(length, coorXY, angleRad, trans, radius, rotationRad):
    angle1 = np.arctan(radius/(length/2)) + rotationRad
    angle2 = np.pi - np.arctan(radius/(length/2)) + rotationRad
    angle3 = np.pi + np.arctan(radius/(length/2)) + rotationRad
    angle4 = 2 * np.pi - np.arctan(radius/(length/2)) + rotationRad

    cirOriginLeft  = (trans[0] - length/2*np.cos(rotationRad), trans[1] - length/2*np.sin(rotationRad))
    cirOriginRight = (trans[0] + length/2*np.cos(rotationRad), trans[1] + length/2*np.sin(rotationRad))
    # myPlot(cirOriginLeft, trans, cirOriginRight, coorXY=coorXY)

    error = 0
    for itr, angle in enumerate(angleRad):
        if angle[0] <= angle1:
            error += abs(((coorXY[itr][0] - cirOriginRight[0])**2 + (coorXY[itr][1] - cirOriginRight[1])**2)**0.5 - radius)
        elif angle[0] < angle2:
            error += abs(((coorXY[itr][0] - trans[0])**2 + (coorXY[itr][1] -trans[1])**2)**0.5 * np.sin(angle[0]-rotationRad) - radius)
        elif angle[0] < angle3:
            error += abs(((coorXY[itr][0] - cirOriginLeft[0])**2 + (coorXY[itr][1] - cirOriginLeft[1])**2)**0.5 - radius)
        else:
            error += abs(-((coorXY[itr][0] - trans[0])**2 + (coorXY[itr][1] -trans[1])**2)**0.5 * np.sin(angle[0]-rotationRad) - radius)

    num = len(angleRad)
    cost = error/num

    return cost


# Main
if __name__ == '__main__':
    transX     = 1000
    transY     = 800
    length      = 80
    radius      = 100
    rotationDeg = 30

    coorXY = DrawSlotPts(transX=transX, transY=transY, length=length, radius=radius, rotationDeg=rotationDeg)

    # Guess slot's properties
    trans = FindTranslations(coorXY)
    rotationRad = FindSlotRotation(coorXY, trans)
    angleRad = FindPtsAngle(coorXY, trans, rotationRad)
    length, radius = FindLengthRadius(coorXY, trans, angleRad, rotationRad)
    print(trans, length, radius, rotationRad)

    cost = GetCost(length, coorXY, angleRad, trans, radius, rotationRad)
    print(cost)

    # Optimization
    # x0 = np.array([length, ])
    # result = scipy.optimize.minimize(fun=GetCost, x0=x0, args=(coorXY, angleRad, trans, radius, rotationRad))
    # print(result.x)
