import numpy as np
import matplotlib.pyplot as plt
import random


#####################################################################################
# Create testing data
#####################################################################################
def CreateSlotPts(transX, transY, length, radius, rotationDeg):
    directionDeg = np.linspace(0, 360, 31)
    directionRad = 2* np.pi / 360 * directionDeg

    coorX = [radius*(1 + random.random()/10000) * np.cos(rad) for rad in directionRad]
    coorY = [radius*(1 + random.random()/10000) * np.sin(rad) for rad in directionRad]

    coorX = [coor-length/2 if coor<0 else coor+length/2 for coor in coorX]

    coorXY = []
    rotationRad = 2*np.pi / 360 * rotationDeg
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
            thetaItr = 2*np.pi - artTan

        coorX[itr] = radiusItr * np.cos(rotationRad+thetaItr) + transX
        coorY[itr] = radiusItr * np.sin(rotationRad+thetaItr) + transY

        coorXY.append((coorX[itr], coorY[itr]))
    
    return tuple(coorXY)


#####################################################################################
# Plot function
#####################################################################################
def geoPlot(*points, fig, ax, dataSet):
    if fig == None and ax == None:
        fig = plt.figure(figsize=(5,5), tight_layout=True)
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('coorX')
        ax.set_ylabel('coorY')
    
    if dataSet:
        dataX = [data[0] for data in dataSet]
        dataY = [data[1] for data in dataSet]
        ax.plot(dataX, dataY, '-o', markersize=4, markeredgewidth=1, markerfacecolor='w', markeredgecolor='royalblue')

    if points:
        for point in points:
            ax.plot(point[0], point[1], '+', markersize=10, markeredgewidth=2, markerfacecolor='orange', markeredgecolor='orange')
            ax.grid()
    
    return fig, ax


def costPlot(dataSet):
    dataX = [data[0] for data in dataSet]
    dataY = [data[1] for data in dataSet]

    fig = plt.figure(figsize=(5,3), tight_layout=True)
    ax = fig.add_subplot(1,1,1)
    ax.plot(dataX, dataY, '-', color='royalblue', linewidth=3)
    ax.set_xlabel('iterations')
    ax.set_ylabel('cost')
    ax.grid()
    plt.show()


def printGuessResult(itr, cost, trans, slotRotation, length, radius):
    print('Guess no. : {:4d}'.format(itr))
    print('  - cost  : {:.6f}'.format(cost))
    print('  - transX: {:.6f}'.format(trans[0]))
    print('  - transY: {:.6f}'.format(trans[1]))
    print('  - rotate: {:.6f}'.format(slotRotation*360/(2*np.pi)))
    print('  - length: {:.6f}'.format(length))
    print('  - radius: {:.6f}'.format(radius))


def GetGuessedCornerCoor(trans, slotRotation, length, radius):
    diagLength = ((length/2)**2 + radius**2)**0.5
    diagTheta1 = np.arctan(radius/(length/2))
    diagTheta2 = np.pi - np.arctan(radius/(length/2))

    ptLT = (trans[0] + diagLength*np.cos(diagTheta2+slotRotation) , trans[1] + diagLength*np.sin(diagTheta2+slotRotation))
    ptRT = (trans[0] + diagLength*np.cos(diagTheta1+slotRotation) , trans[1] + diagLength*np.sin(diagTheta1+slotRotation))
    ptLB = (trans[0] - diagLength*np.cos(diagTheta1+slotRotation) , trans[1] - diagLength*np.sin(diagTheta1+slotRotation))
    ptRB = (trans[0] - diagLength*np.cos(diagTheta2+slotRotation) , trans[1] - diagLength*np.sin(diagTheta2+slotRotation))

    return ptLT, ptRT, ptLB, ptRB


#####################################################################################
# Hypothesis model
#####################################################################################
def FindTranslations(coorXY):
    maxX = max([coor[0] for coor in coorXY])
    maxY = max([coor[1] for coor in coorXY])
    minX = min([coor[0] for coor in coorXY])
    minY = min([coor[1] for coor in coorXY])

    return ((maxX+minX)/2, (maxY+minY)/2)


def FindSlotRotation(coorXY, trans):
    distance = [GetDistance(coor, trans) for coor in coorXY]
    
    maxDist = 0
    for itr, dist in enumerate(distance):
        if dist > maxDist and (coorXY[itr][0]-trans[0]) > 0:
            maxId = itr
            maxDist = dist
    
    deltaX = coorXY[maxId][0] - trans[0]
    deltaY = coorXY[maxId][1] - trans[1]

    if deltaY > 0:
        slotRotation = np.arctan(abs(deltaY)/abs(deltaX))
    else:
        slotRotation = 2*np.pi - np.arctan(abs(deltaY)/abs(deltaX))

    return slotRotation


def FindPtsDirection(coorXY, trans):
    # ptsDirection should be "absolute" not "relative" 
    ptsDirection = []
    for coor in coorXY:
        deltaX = coor[0] - trans[0]
        deltaY = coor[1] - trans[1]

        if deltaX >=0 and deltaY >= 0:
            direction = np.arctan(abs(deltaY)/abs(deltaX))
        elif deltaX < 0 and deltaY >=0:
            direction = np.pi - np.arctan(abs(deltaY)/abs(deltaX))
        elif deltaX < 0 and deltaY < 0:
            direction = np.pi + np.arctan(abs(deltaY)/abs(deltaX))
        else:
            direction = 2*np.pi - np.arctan(abs(deltaY)/abs(deltaX))

        if direction > 2*np.pi:
            direction -= 2*np.pi

        ptsDirection.append(direction)
    
    return ptsDirection


def FindSlotLengthRadius(coorXY, trans, ptsDirection, slotRotation):
    distance = [GetDistance(coor, trans) for coor in coorXY]
    maxDist = max(distance)
    yProject = [dist * abs(np.sin(ptsDirection[itr]-slotRotation))  for itr, dist in enumerate(distance)]

    radius = max(yProject)
    length = 2 * (maxDist - radius)

    return length, radius

        
def GetCost(coorXY, ptsDirection, trans, slotRotation, length, radius):
    direction1 = np.arctan(radius/(length/2)) + slotRotation
    direction2 = np.pi - np.arctan(radius/(length/2)) + slotRotation
    direction3 = np.pi + np.arctan(radius/(length/2)) + slotRotation
    direction4 = 2*np.pi - np.arctan(radius/(length/2)) + slotRotation

    cirOriginLeft  = (trans[0] - length/2*np.cos(slotRotation), trans[1] - length/2*np.sin(slotRotation))
    cirOriginRight = (trans[0] + length/2*np.cos(slotRotation), trans[1] + length/2*np.sin(slotRotation))

    error = 0
    for itr, direction in enumerate(ptsDirection):
        if direction <= direction1:
            error += abs(GetDistance(coorXY[itr], cirOriginRight) - radius)
        elif direction < direction2:
            error += abs(GetDistance(coorXY[itr], trans)*np.sin(direction-slotRotation) - radius)
        elif direction < direction3:
            error += abs(GetDistance(coorXY[itr], cirOriginLeft) - radius)
        elif direction < direction4:
            error += abs(-GetDistance(coorXY[itr], trans)*np.sin(direction-slotRotation) - radius)
        else:
            error += abs(GetDistance(coorXY[itr], cirOriginRight) - radius)

    num = len(ptsDirection)
    cost = error/num
    return cost


def GetDistance(pt1, pt2):
    deltaX = pt2[0] - pt1[0]
    deltaY = pt2[1] - pt1[1]
    return (deltaX**2 + deltaY**2)**0.5


def GetGradient(func, tgt, *arguments):
    coorXY, ptsDirection, trans, slotRotation, length, radius = arguments
    
    if tgt == 'transX':
        argumentLeft = trans[0] * 0.999
        argumentRight = trans[0] * 1.001

        costLeft = func(coorXY, ptsDirection, (argumentLeft, trans[1]), slotRotation, length, radius)
        costRight = func(coorXY, ptsDirection, (argumentRight, trans[1]), slotRotation, length, radius)
    
    elif tgt == 'transY':
        argumentLeft = trans[1] * 0.999
        argumentRight = trans[1] * 1.001

        costLeft = func(coorXY, ptsDirection, (trans[0], argumentLeft), slotRotation, length, radius)
        costRight = func(coorXY, ptsDirection, (trans[0], argumentRight), slotRotation, length, radius)

    elif tgt == 'slotRotation':
        argumentLeft = slotRotation * 0.999
        argumentRight = slotRotation * 1.001

        costLeft = func(coorXY, ptsDirection, trans, argumentLeft, length, radius)
        costRight = func(coorXY, ptsDirection, trans, argumentRight, length, radius)

    elif tgt == 'length':
        argumentLeft = length * 0.999
        argumentRight = length * 1.001

        costLeft = func(coorXY, ptsDirection, trans, slotRotation, argumentLeft, radius)
        costRight = func(coorXY, ptsDirection, trans, slotRotation, argumentRight, radius)
    
    elif tgt == 'radius':
        argumentLeft = radius * 0.999
        argumentRight = radius * 1.001

        costLeft = func(coorXY, ptsDirection, trans, slotRotation, length, argumentLeft)
        costRight = func(coorXY, ptsDirection, trans, slotRotation, length, argumentRight)
    
    gradient = (costRight-costLeft) / (argumentRight-argumentLeft)
    return gradient


#####################################################################################
# Main
#####################################################################################
def main(coorXY, printData=False):
    # A precise initial guesses of slot's properties
    trans = FindTranslations(coorXY)
    slotRotation = FindSlotRotation(coorXY, trans)
    ptsDirection = FindPtsDirection(coorXY, trans)
    length, radius = FindSlotLengthRadius(coorXY, trans, ptsDirection, slotRotation)
    alpha, maxItr, costRateUpperBound = 0.0001, 1000, 0.0001
    
    # A coarse initial guesses of slot's properties
    # trans = (990, 820)
    # slotRotation = rotationDeg * 2*np.pi / 360
    # ptsDirection = FindPtsDirection(coorXY, trans)
    # length, radius = 50, 120
    # alpha, maxItr, costRateUpperBound = 0.1, 1000, 0.000001

    cost = GetCost(coorXY, ptsDirection, trans, slotRotation, length, radius)
    if printData:
        printGuessResult(0, cost, trans, slotRotation, length, radius)
        fig, ax = geoPlot(fig=None, ax=None, dataSet=coorXY)

    # Optimize guesses: trans, slotRotation, length, radius
    itr = 1
    costRecord = [(0, cost)]
    while itr <= maxItr:
        grad = GetGradient(GetCost, 'transX', coorXY, ptsDirection, trans, slotRotation, length, radius)
        trans = (trans[0]-alpha*grad, trans[1]           )
        
        grad = GetGradient(GetCost, 'transY', coorXY, ptsDirection, trans, slotRotation, length, radius)
        trans = (trans[0]           , trans[1]-alpha*grad)
        
        grad = GetGradient(GetCost, 'slotRotation', coorXY, ptsDirection, trans, slotRotation, length, radius)
        slotRotation += -alpha*grad

        grad = GetGradient(GetCost, 'length', coorXY, ptsDirection, trans, slotRotation, length, radius)
        length += -alpha*grad
        
        grad = GetGradient(GetCost, 'radius', coorXY, ptsDirection, trans, slotRotation, length, radius)
        radius += -alpha*grad

        if printData:
            printGuessResult(itr, cost, trans, slotRotation, length, radius)
            ptLT, ptRT, ptLB, ptRB = GetGuessedCornerCoor(trans, slotRotation, length, radius)
            fig, ax = geoPlot(ptLT, ptRT, ptLB, ptRB, fig=fig, ax=ax, dataSet=None)

        newCost = GetCost(coorXY, ptsDirection, trans, slotRotation, length, radius)
        costRecord.append((itr, newCost))
        itr += 1

        costRate = abs((newCost-cost) / cost)
        if costRate < costRateUpperBound:
            break
        else:
            cost = newCost
    
    if printData:
        costPlot(costRecord)
        
    return trans, slotRotation, length, radius


# Test
if __name__ == '__main__':
    transX     = 1000
    transY     = 800
    length      = 60
    radius      = 100
    rotationDeg = 45

    coorXY = CreateSlotPts(transX=transX, transY=transY, length=length, radius=radius, rotationDeg=rotationDeg)
    transOpt, rotationRadOpt, lengthOpt, radiusOpt = main(coorXY, printData=True)