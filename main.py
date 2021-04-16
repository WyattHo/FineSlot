import numpy as np
import matplotlib.pyplot as plt
import random


#####################################################################################
# Create testing data
#####################################################################################
def DrawSlotPts(transX, transY, length, radius, rotationDeg):
    angleDeg = np.linspace(0, 360, 31)
    angleRad = 2* np.pi / 360 * angleDeg

    coorX = [radius*(1 + random.random()/10000) * np.cos(rad) for rad in angleRad]
    coorY = [radius*(1 + random.random()/10000) * np.sin(rad) for rad in angleRad]

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


#####################################################################################
# Hypothesis model
#####################################################################################
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

        angleRad.append(theta)
    
    return angleRad

        
def GetCost(coorXY, angleRad, trans, rotationRad, length, radius):
    angle1 = np.arctan(radius/(length/2)) + rotationRad
    angle2 = np.pi - np.arctan(radius/(length/2)) + rotationRad
    angle3 = np.pi + np.arctan(radius/(length/2)) + rotationRad
    # angle4 = 2 * np.pi - np.arctan(radius/(length/2)) + rotationRad

    cirOriginLeft  = (trans[0] - length/2*np.cos(rotationRad), trans[1] - length/2*np.sin(rotationRad))
    cirOriginRight = (trans[0] + length/2*np.cos(rotationRad), trans[1] + length/2*np.sin(rotationRad))
    # myPlot(cirOriginLeft, trans, cirOriginRight, coorXY=coorXY)

    error = 0
    for itr, angle in enumerate(angleRad):
        if angle <= angle1:
            error += abs(((coorXY[itr][0] - cirOriginRight[0])**2 + (coorXY[itr][1] - cirOriginRight[1])**2)**0.5 - radius)
        elif angle < angle2:
            error += abs(((coorXY[itr][0] - trans[0])**2 + (coorXY[itr][1] -trans[1])**2)**0.5 * np.sin(angle-rotationRad) - radius)
        elif angle < angle3:
            error += abs(((coorXY[itr][0] - cirOriginLeft[0])**2 + (coorXY[itr][1] - cirOriginLeft[1])**2)**0.5 - radius)
        else:
            error += abs(-((coorXY[itr][0] - trans[0])**2 + (coorXY[itr][1] -trans[1])**2)**0.5 * np.sin(angle-rotationRad) - radius)

    num = len(angleRad)
    cost = error/num

    return cost


def GetGradient(func, tgt, *arguments):
    coorXY, angleRad, trans, rotationRad, length, radius = arguments
    
    if tgt == 'transX':
        argumentLeft = trans[0] * 0.99999
        argumentRight = trans[0] * 1.00001

        costLeft = func(coorXY, angleRad, (argumentLeft, trans[1]), rotationRad, length, radius)
        costRight = func(coorXY, angleRad, (argumentRight, trans[1]), rotationRad, length, radius)
    
    elif tgt == 'transY':
        argumentLeft = trans[1] * 0.99999
        argumentRight = trans[1] * 1.00001

        costLeft = func(coorXY, angleRad, (trans[0], argumentLeft), rotationRad, length, radius)
        costRight = func(coorXY, angleRad, (trans[0], argumentRight), rotationRad, length, radius)

    elif tgt == 'rotationRad':
        argumentLeft = rotationRad * 0.999999
        argumentRight = rotationRad * 1.000001

        costLeft = func(coorXY, angleRad, trans, argumentLeft, length, radius)
        costRight = func(coorXY, angleRad, trans, argumentRight, length, radius)

    elif tgt == 'length':
        argumentLeft = length * 0.999
        argumentRight = length * 1.001

        costLeft = func(coorXY, angleRad, trans, rotationRad, argumentLeft, radius)
        costRight = func(coorXY, angleRad, trans, rotationRad, argumentRight, radius)
    
    elif tgt == 'radius':
        argumentLeft = radius * 0.999
        argumentRight = radius * 1.001

        costLeft = func(coorXY, angleRad, trans, rotationRad, length, argumentLeft)
        costRight = func(coorXY, angleRad, trans, rotationRad, length, argumentRight)
    

    gradient = (costRight-costLeft) / (argumentRight-argumentLeft)

    return gradient


# Main
if __name__ == '__main__':
    transX     = 1000
    transY     = 800
    length      = 80
    radius      = 100
    rotationDeg = 30 

    coorXY = DrawSlotPts(transX=transX, transY=transY, length=length, radius=radius, rotationDeg=rotationDeg)

    # A precise initial guesses of slot's properties
    trans = FindTranslations(coorXY)
    rotationRad = FindSlotRotation(coorXY, trans)
    angleRad = FindPtsAngle(coorXY, trans, rotationRad)
    length, radius = FindLengthRadius(coorXY, trans, angleRad, rotationRad)
    
    # A coarse initial guesses of slot's properties
    # trans = (990, 810)
    # rotationRad = 0.5236
    # angleRad = FindPtsAngle(coorXY, trans, rotationRad)
    # length, radius = 84, 105

    cost = GetCost(coorXY, angleRad, trans, rotationRad, length, radius)
    print('Cost of initial: {:.6f}'.format(cost))
    print('  - transX: {:.6f}'.format(trans[0]))
    print('  - transY: {:.6f}'.format(trans[1]))
    print('  - rotate: {:.6f}'.format(rotationRad))
    print('  - length: {:.6f}'.format(length))
    print('  - radius: {:.6f}'.format(radius))
    print('\n')


    # Optimize guesses: trans, rotationRad, length, radius
    alpha = 4
    itr = 1
    maxItr = 30

    examCost = [(0, cost)]
    while itr <= maxItr:
        transXGrad = GetGradient(GetCost, 'transX', coorXY, angleRad, trans, rotationRad, length, radius)
        trans = (trans[0] - alpha * transXGrad, trans[1])
        
        transYGrad = GetGradient(GetCost, 'transY', coorXY, angleRad, trans, rotationRad, length, radius)
        trans = (trans[0], trans[1] - alpha * transYGrad)
        
        # rotationRadGrad = GetGradient(GetCost, 'rotationRad', coorXY, angleRad, trans, rotationRad, length, radius)
        # rotationRad += -alpha * rotationRadGrad

        lengthGrad = GetGradient(GetCost, 'length', coorXY, angleRad, trans, rotationRad, length, radius)
        length += -alpha * lengthGrad
        
        radiusGrad = GetGradient(GetCost, 'radius', coorXY, angleRad, trans, rotationRad, length, radius)
        radius += -alpha * radiusGrad

        newCost = GetCost(coorXY, angleRad, trans, rotationRad, length, radius)
        print('Cost of itr {:2d}: {:.6f}'.format(itr, newCost))
        print('  - transX: {:.6f}'.format(trans[0]))
        print('  - transY: {:.6f}'.format(trans[1]))
        print('  - rotate: {:.6f}'.format(rotationRad))
        print('  - length: {:.6f}'.format(length))
        print('  - radius: {:.6f}'.format(radius))
        print('\n')
        
        examCost.append((itr, newCost))
        itr += 1

        costRate = abs((newCost - cost)/cost)

        if costRate < 0.01:
            break
        else:
            cost = newCost

    myPlot(coorXY=examCost)
