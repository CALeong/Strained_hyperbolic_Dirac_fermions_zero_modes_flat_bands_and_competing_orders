import numpy as np
from Fundamental.Number_Points import points #Code for getting points per level and total number points


# p = 8 #p value for hyperbolic lattice
# q = 4 #q value for hyperbolic lattice
# num_levels = 4 #Number of levels to consider

def H0(p,q,num_levels):

    points_perlevel, totalnum = points(p,q,num_levels) #Get points per level and total number of points
    # print(points_perlevel)
    # print(totalnum)

    #Code to generate list of cumulative points
    #ie if level 1 has 10 points and level two has 20 points
    #Then cumulative list: 10, 30
    point_cumulative = np.array([],dtype=int)
    for a in range(len(points_perlevel)):
        if a == 0:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[0]))
        elif a == len(points_perlevel)-1:
            point_cumulative = np.append(point_cumulative, int(totalnum))
        else:
            point_cumulative = np.append(point_cumulative,int(points_perlevel[a]+point_cumulative[-1]))

    # print(point_cumulative)

    #Initialize H0 matrix (original coordinates)
    H0 = np.zeros((int(totalnum),int(totalnum)),dtype=int)

    #Function to set often used counter variables to zero
    #Used before for loop code for populating H0 for each level
    def reinitialize_counter():
        threecounter = 0
        fourcounter = 0
        counter = 0
        subcounter = 0
        onsub = False
        return(threecounter,fourcounter,counter,subcounter,onsub)

    if q == 3: #q=3 H0 populating code
        threecounter,fourcounter,counter,subcounter, onsub = reinitialize_counter()
        if num_levels >= 1:
            for n in range(point_cumulative[0]): #First Level
                if n == 0: # First point on level
                    H0[n][n+1] = 1 #Connected to point next to it
                    H0[n+1][n] = 1
                    if num_levels > 1:
                        H0[n][point_cumulative[0]] = 1
                        H0[point_cumulative[0]][n] = 1
                elif n == point_cumulative[0]-1: #Last point on level
                    H0[n][0] = 1 #Connected to first point in level
                    H0[0][n] = 1
                    #n * (p-q) is because each point connected to level one point is separated by (p-q) points
                    #Thus point_cumulative[0] + n * (p - q) gives value for the point connected to nth first level point
                    if num_levels > 1:
                        H0[n][point_cumulative[0] + n * (p - q)] = 1 #Code for finding connections with next level
                        H0[point_cumulative[0] + n * (p - q)][n] = 1
                else: #All other points
                    H0[n][n + 1] = 1 #Connected to next point on first level
                    H0[n + 1][n] = 1
                    H0[n][n - 1] = 1 #Connected to previous point on first level
                    H0[n - 1][n] = 1
                    if num_levels > 1:
                        H0[n][point_cumulative[0]+ n*(p-q)] = 1 #Same deal as above with connecting to points on next level
                        H0[point_cumulative[0]+ n*(p-q)][n] = 1
        threecounter,fourcounter,counter,subcounter, onsub = reinitialize_counter()
        if num_levels >= 2:
            for n in range(point_cumulative[0],point_cumulative[1]): #Second Level
                if n == point_cumulative[0]: #First point
                    H0[n][n + 1] = 1 #Point next to it is connected
                    H0[n + 1][n] = 1 #Dont worry about previous point as that is handled in "Last Point" code
                    counter += 1 #Add one to counter to start counting (relevant in later code)
                elif n == point_cumulative[1]-1: #Last Point
                    H0[n][point_cumulative[0]] = 1 #Connect last point with first point on second level
                    H0[point_cumulative[0]][n] = 1
                    #Fourcounter and three counter explained later in code
                    #Connect last point of level with points on next level
                    if num_levels != 2:
                        H0[n][point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)] = 1
                        H0[point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)][n] = 1
                elif counter == 0: #Points where no connected points on next level
                    counter += 1
                    H0[n][n + 1] = 1 #Connect with next point on level
                    H0[n + 1][n] = 1
                    H0[n][n - 1] = 1 #Connect with previous point on level
                    H0[n - 1][n] = 1
                    fourcounter += 1 #Add fourcounter since counter==0 point occurs at four-border polygon
                elif counter != (p-q-1): #Points where there is a connected point on next level
                    counter += 1
                    H0[n][n + 1] = 1 #Connect with next point on level
                    H0[n + 1][n] = 1
                    H0[n][n - 1] = 1 #Connect with previous point on level
                    H0[n - 1][n] = 1
                    #fourcounter = number of four border polygons passed
                    #threecounter = number of three border polygons passed
                    #n-border polygon means polygon with n sides that do not have next level points on both ends of edge
                    #(p-q-1) next level points on four border polygon
                    #(p-q) next level points on three border polygon
                    #Calculation thus gives point on next level connected to point on second level
                    if num_levels != 2:
                        H0[n][point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)] = 1
                        H0[point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)][n] = 1
                        threecounter += 1 #Add threecounter since each point traverses a three border polygon unless counter==0
                elif counter == (p-q-1): #Last point that is connected with next level before counter==0 point occurs
                    counter = 0 #set counter=0 so code knows we are at four border polygon point
                    H0[n][n + 1] = 1 #Connect point with next point on label
                    H0[n + 1][n] = 1
                    H0[n][n - 1] = 1 #Connect point with last point on label
                    H0[n - 1][n] = 1
                    #Same deal as before with getting point on next level connected to second level point
                    if num_levels != 2:
                        H0[n][point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)] = 1
                        H0[point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)][n] = 1

        threecounter,fourcounter,counter,subcounter, onsub = reinitialize_counter()
        onsubcounter = 0
        if num_levels >= 3:
            for n in range(point_cumulative[1],point_cumulative[2]): #Third Level
                if n == point_cumulative[1]: #First point on third level
                    H0[n][n + 1] = 1 #Connect point with next point on third level
                    H0[n + 1][n] = 1
                    counter += 1 #Start counter for later
                    H0[n][point_cumulative[2]-1] = 1
                    H0[point_cumulative[2]-1][n] = 1
                elif n == point_cumulative[2]-1:
                    threecounter += 1
                    H0[n][point_cumulative[1]] = 1
                    H0[point_cumulative[1]][n] = 1
                    if num_levels != 3:
                        H0[n][point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)] = 1
                        H0[point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)][n] = 1
                elif subcounter == (p-q-2):
                    H0[n][n+1] = 1
                    H0[n+1][n] = 1
                    onsubcounter += 1
                    if onsubcounter == 1:
                        if num_levels != 3:
                            H0[n][point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)] = 1
                            H0[point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)][n] = 1
                    elif onsubcounter == (p-q-1):
                        counter = 1
                        onsubcounter = 0
                        subcounter = 0
                        fourcounter += 1
                    else:
                        threecounter += 1
                        if num_levels != 3:
                            H0[n][point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)] = 1
                            H0[point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)][n] = 1
                elif counter == 1:
                    H0[n][n + 1] = 1
                    H0[n + 1][n] = 1
                    if num_levels != 3:
                        H0[n][point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)] = 1
                        H0[point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)][n] = 1
                    counter += 1
                elif counter != (p-q-1) and counter != 0:
                    threecounter += 1
                    H0[n][n+1] = 1
                    H0[n+1][n] = 1
                    if num_levels != 3:
                        H0[n][point_cumulative[2]+threecounter*(p-q)+fourcounter*(p-q-1)] = 1
                        H0[point_cumulative[2]+threecounter*(p-q)+fourcounter*(p-q-1)][n] = 1
                    counter += 1
                elif counter == (p-q-1):
                    counter = 0
                    threecounter += 1
                    H0[n][n + 1] = 1
                    H0[n + 1][n] = 1
                    if num_levels != 3:
                        H0[n][point_cumulative[2] + threecounter * (p - q) + fourcounter*(p-q-1)] = 1
                        H0[point_cumulative[2] + threecounter * (p - q) + fourcounter*(p-q-1)][n] = 1
                elif counter == 0:
                    counter += 1
                    H0[n][n + 1] = 1
                    H0[n + 1][n] = 1
                    fourcounter += 1
                    subcounter += 1
        threecounter,fourcounter,counter,subcounter, onsub = reinitialize_counter()
        onsubcounter = 0
        threecount = (p-q) #Number points traversed on next level when polygon on fourth level is three border
        fourcount = (p-q-1) #Number points traversed on next level when polygon on fourth level is four border
        subsubcounter = 0
        #Two types of motifs in q=3 fourth level
        #One is (p-q-2) cycles of the counter then one subcounter
        #Other motif is (p-q-3) cycles of counter followed by subcounter
        motif1 = True
        motif2 = False
        if num_levels >= 4:
            for n in range(point_cumulative[2],point_cumulative[3]): #Fourth Level
                if n == point_cumulative[2]: #First point
                    H0[n+1][n] = 1
                    H0[n][n+1] = 1
                    H0[n][point_cumulative[3]-1] = 1
                    H0[point_cumulative[3]-1][n] = 1
                    counter += 1
                elif n == point_cumulative[3]-1: #Last point
                    H0[n - 1][n] = 1
                    H0[n][n - 1] = 1
                    threecounter += 1
                    if num_levels>4:
                        H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = 1
                        H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = 1
                elif motif2 == True: #On part of lattice that is motif2 (described above)
                    if subcounter == (p-q-3): #Means now on subcounter portion at end of motif2
                        onsubcounter += 1 #counter to bring code through subcounter
                        if onsubcounter == 1:
                            H0[n][n+1] = 1
                            H0[n+1][n] = 1
                            H0[n][n - 1] = 1
                            H0[n - 1][n] = 1
                            if num_levels>4:
                                H0[n][point_cumulative[3]+fourcount*fourcounter+threecount*threecounter] = 1
                                H0[point_cumulative[3]+fourcount*fourcounter+threecount*threecounter][n] = 1
                        elif onsubcounter != (p-q-1):
                            H0[n][n + 1] = 1
                            H0[n + 1][n] = 1
                            H0[n][n - 1] = 1
                            H0[n - 1][n] = 1
                            threecounter += 1
                            if num_levels>4:
                                H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = 1
                                H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = 1
                        elif onsubcounter == (p-q-1): #Last point in subcounter, reset values so they are ready for next point
                            fourcounter += 1
                            onsubcounter = 0
                            subcounter = 0
                            subsubcounter = 0
                            counter = 1
                            H0[n][n + 1] = 1
                            H0[n + 1][n] = 1
                            H0[n][n - 1] = 1
                            H0[n - 1][n] = 1
                            motif1 = True #Switch back to motif1
                            motif2 = False
                    elif counter == 1: #Cycle through counter portions
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        H0[n][n - 1] = 1
                        H0[n - 1][n] = 1
                        if num_levels>4:
                            H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = 1
                            H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = 1
                        counter += 1
                    elif counter != (p-q):
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        H0[n][n - 1] = 1
                        H0[n - 1][n] = 1
                        threecounter += 1
                        if num_levels>4:
                            H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = 1
                            H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = 1
                        counter += 1
                    elif counter == (p-q):
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        H0[n][n - 1] = 1
                        H0[n - 1][n] = 1
                        fourcounter += 1
                        subcounter += 1 #Add to subcounter. Once threshold reached, tells code that it is on subcounter portion
                        counter = 1
                elif motif1 == True: #motif1 protion of lattice described above
                    if subcounter == (p-q-2): #subcounter portion of motif1
                        onsubcounter += 1 #cycle through subcounter portion
                        if onsubcounter == 1:
                            H0[n][n+1] = 1
                            H0[n+1][n] = 1
                            H0[n][n - 1] = 1
                            H0[n - 1][n] = 1
                            if num_levels>4:
                                H0[n][point_cumulative[3]+fourcount*fourcounter+threecount*threecounter] = 1
                                H0[point_cumulative[3]+fourcount*fourcounter+threecount*threecounter][n] = 1
                        elif onsubcounter != (p-q-1):
                            H0[n][n + 1] = 1
                            H0[n + 1][n] = 1
                            H0[n][n - 1] = 1
                            H0[n - 1][n] = 1
                            threecounter += 1
                            if num_levels>4:
                                H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = 1
                                H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = 1
                        elif onsubcounter == (p-q-1): #Last point, reset values for next point
                            fourcounter += 1
                            onsubcounter = 0
                            subcounter = 0
                            counter = 1
                            H0[n][n + 1] = 1
                            H0[n + 1][n] = 1
                            H0[n][n - 1] = 1
                            H0[n - 1][n] = 1
                            subsubcounter += 1
                            if subsubcounter == (p-q-2): #If certain number of subcounters passed on motif1, need to switch to motif2
                                motif1 = False
                                motif2 = True
                    elif counter == 1: #cycle through counter portion of motif1 (with breaks to subcounter when at those points)
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        H0[n][n - 1] = 1
                        H0[n - 1][n] = 1
                        if num_levels>4:
                            H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = 1
                            H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = 1
                        counter += 1
                    elif counter != (p-q):
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        H0[n][n - 1] = 1
                        H0[n - 1][n] = 1
                        threecounter += 1
                        if num_levels>4:
                            H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = 1
                            H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = 1
                        counter += 1
                    elif counter == (p-q):
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        H0[n][n - 1] = 1
                        H0[n - 1][n] = 1
                        fourcounter += 1
                        subcounter += 1 #counter to keep track so code knows when its on subcounter of motif1
                        counter = 1

        if num_levels == 5:
            for n in range(point_cumulative[3],point_cumulative[4]):
                if n == point_cumulative[4]-1:
                    H0[n][point_cumulative[3]] = 1
                    H0[point_cumulative[3]][n] = 1
                else:
                    H0[n][n+1] = 1
                    H0[n+1][n] = 1



    if q == 4: #q=4 H0 populating code
        threecounter,fourcounter,counter,subcounter,onsub = reinitialize_counter()
        oddcounter = 0
        oddcount = (p-q+1)
        evencounter = 0
        evencount = 1
        for n in range(point_cumulative[0]): #First level
            if n == 0:
                H0[n][n+1] = 1
                H0[n+1][n] = 1
                H0[n][point_cumulative[0]-1] = 1
                H0[point_cumulative[0]-1][n] = 1
                H0[n][point_cumulative[0]] = 1
                H0[point_cumulative[0]][n] = 1
                H0[n][point_cumulative[1]-1] = 1
                H0[point_cumulative[1]-1][n] = 1
                counter += 1
            elif n == point_cumulative[0]-1:
                evencounter += 1
                oddcounter += 1
                H0[n][n - 1] = 1
                H0[n - 1][n] = 1
                H0[n][point_cumulative[0] + oddcounter * oddcount + evencounter * evencount] = 1
                H0[point_cumulative[0] + oddcounter * oddcount + evencounter * evencount][n] = 1
                H0[point_cumulative[0] + oddcounter * oddcount + evencounter * evencount - 1][n] = 1
                H0[n][point_cumulative[0] + oddcounter * oddcount + evencounter * evencount - 1] = 1
            else:
                evencounter += 1
                oddcounter += 1
                H0[n][n + 1] = 1
                H0[n + 1][n] = 1
                H0[n][n - 1] = 1
                H0[n - 1][n] = 1
                H0[n][point_cumulative[0]+oddcounter*oddcount+evencounter*evencount] = 1
                H0[point_cumulative[0]+oddcounter*oddcount+evencounter*evencount][n] = 1
                H0[n][point_cumulative[0]+oddcounter*oddcount+evencounter*evencount-1] = 1
                H0[point_cumulative[0]+oddcounter*oddcount+evencounter*evencount-1][n] = 1

        threecounter, fourcounter, counter, subcounter, onsub = reinitialize_counter()
        oddcounter = 0
        oddcount = (p - q + 1)
        evencounter = 0
        evencount = 1
        if num_levels > 1:
            for n in range(point_cumulative[0],point_cumulative[1]):
                if n == point_cumulative[0]:
                    H0[n][n+1] = 1
                    H0[n+1][n] = 1
                    if num_levels > 2:
                        H0[n][point_cumulative[2]-1] = 1
                        H0[point_cumulative[2]-1][n] = 1
                        H0[n][point_cumulative[1]] = 1
                        H0[point_cumulative[1]][n] = 1
                    counter += 1
                elif counter == 0:
                    H0[n][n+1] = 1
                    H0[n+1][n] = 1
                    if num_levels > 2:
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount] = 1
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount][n] = 1
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1] = 1
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1][n] = 1
                    counter += 1
                elif counter == (p-q+1):
                    oddcounter += 1
                    evencounter += 1
                    H0[n][n - 1] = 1
                    H0[n - 1][n] = 1
                    if num_levels > 2:
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount] = 1
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount][n] = 1
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1] = 1
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1][n] = 1
                    counter = 0
                    oddcounter += 1
                elif counter != (p-q+1):
                    oddcounter += 1
                    evencounter += 1
                    H0[n][n + 1] = 1
                    H0[n + 1][n] = 1
                    H0[n][n - 1] = 1
                    H0[n - 1][n] = 1
                    if num_levels > 2:
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount] = 1
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount][n] = 1
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1] = 1
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1][n] = 1
                    counter += 1
        threecounter, fourcounter, counter, subcounter, onsub = reinitialize_counter()
        oddcounter = 0
        oddcount = (p - q + 1)
        evencounter = 0
        evencount = 1
        subsubcounter = 0
        onsubcounter = 0
        if num_levels > 2:
            for n in range(point_cumulative[1],point_cumulative[2]): #Third Level
                if n == point_cumulative[1]:
                    H0[n][n + 1] = 1
                    H0[n + 1][n] = 1
                    if num_levels > 3:
                        H0[n][point_cumulative[2]] = 1
                        H0[point_cumulative[2]][n] = 1
                        H0[n][point_cumulative[3]-1] = 1
                        H0[point_cumulative[3]-1][n] = 1
                    counter += 1
                elif subcounter == (p-q+1):
                    if onsubcounter == 0:
                        oddcounter += 1
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        if num_levels > 3:
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = 1
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = 1
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = 1
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = 1
                        onsubcounter += 1
                    elif onsubcounter == (p-q+1):
                        onsubcounter = 0
                        subcounter = 0
                        counter = 1
                        oddcounter += 1
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        if num_levels > 3:
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = 1
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = 1
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = 1
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = 1
                    elif onsubcounter == (p-q):
                        oddcounter += 1
                        evencounter += 1
                        H0[n][n - 1] = 1
                        H0[n - 1][n] = 1
                        if num_levels > 3:
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = 1
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = 1
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = 1
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = 1
                        onsubcounter += 1
                    elif onsubcounter != (p-q+1):
                        oddcounter += 1
                        evencounter += 1
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        H0[n][n - 1] = 1
                        H0[n - 1][n] = 1
                        if num_levels > 3:
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = 1
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = 1
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = 1
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = 1
                        onsubcounter += 1
                elif counter == 0:
                    oddcounter += 1
                    H0[n][n + 1] = 1
                    H0[n + 1][n] = 1
                    if num_levels > 3:
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = 1
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = 1
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = 1
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = 1
                    counter += 1
                elif counter != (p-q+1):
                    oddcounter += 1
                    evencounter += 1
                    H0[n][n+1] = 1
                    H0[n+1][n] = 1
                    H0[n][n - 1] = 1
                    H0[n - 1][n] = 1
                    if num_levels > 3:
                        H0[n][point_cumulative[2]+evencounter*evencount+oddcounter*oddcount] = 1
                        H0[point_cumulative[2]+evencounter*evencount+oddcounter*oddcount][n] = 1
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = 1
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = 1
                    counter += 1
                elif counter == (p-q+1):
                    subcounter += 1
                    oddcounter += 1
                    evencounter += 1
                    H0[n][n - 1] = 1
                    H0[n - 1][n] = 1
                    if num_levels > 3:
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = 1
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = 1
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = 1
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = 1
                    counter = 0

        threecounter, fourcounter, counter, subcounter, onsub = reinitialize_counter()
        oddcounter = 0
        oddcount = (p - q + 1)
        evencounter = 0
        evencount = 1
        subsubcounter = 0
        onsubcounter = 0
        if num_levels > 3:
            for n in range(point_cumulative[2],point_cumulative[3]): #Fourth Level
                if n == point_cumulative[2]:
                    H0[n][n+1] = 1
                    H0[n+1][n] = 1
                    counter += 1
                elif n == point_cumulative[3]-1:
                    H0[n][n-1] = 1
                    H0[n-1][n] = 1
                elif subsubcounter == (p-q+1) and subcounter == (p-q):
                    H0[n][n + 1] = 1
                    H0[n + 1][n] = 1
                    counter = 0
                    subcounter = (p-q+1) #Push it to subcounter routine
                    onsubcounter = 1
                elif subcounter == (p-q+1):
                    if onsubcounter == 0:
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        onsubcounter += 1
                    elif onsubcounter == (p-q):
                        H0[n][n - 1] = 1
                        H0[n - 1][n] = 1
                        onsubcounter = 0
                        counter = 0
                        subcounter = 0
                        subsubcounter += 1
                        if subsubcounter == (p-q+1)+1: #Check if subsub portion passed and if it did then reset subsubcounter
                            subsubcounter = 0
                    elif onsubcounter != (p-q):
                        H0[n][n + 1] = 1
                        H0[n + 1][n] = 1
                        H0[n][n - 1] = 1
                        H0[n - 1][n] = 1
                        onsubcounter += 1
                elif counter == (p-q+1):
                    H0[n][n-1] = 1
                    H0[n-1][n] = 1
                    counter = 0
                    subcounter += 1
                elif counter == 0:
                    H0[n][n + 1] = 1
                    H0[n + 1][n] = 1
                    counter += 1
                elif counter != (p-q+1) and counter != 0:
                    H0[n][n - 1] = 1
                    H0[n - 1][n] = 1
                    H0[n][n + 1] = 1
                    H0[n + 1][n] = 1
                    counter += 1

    if q == 5:
        if num_levels >= 1:
            counter = 0
            for n in range(point_cumulative[0]):
                if n == 0:
                    H0[n][n+1] = 1
                    H0[n+1][n] = 1
                    H0[n][point_cumulative[0]-1] = 1
                    H0[point_cumulative[0]-1][n] = 1
                    H0[n][point_cumulative[0]] = 1
                    H0[point_cumulative[0]][n] = 1
                    H0[n][point_cumulative[0]+1] = 1
                    H0[point_cumulative[0]+1][n] = 1
                    H0[n][point_cumulative[1]-1] = 1
                    H0[point_cumulative[1]-1][n] = 1
                    counter += 1
                elif n == point_cumulative[0]-1:
                    H0[n][n - 1] = 1
                    H0[n - 1][n] = 1
                    H0[n][point_cumulative[0] + counter * (p - 3 + 2)] = 1
                    H0[point_cumulative[0] + counter * (p - 3 + 2)][n] = 1
                    H0[n][point_cumulative[0] + counter * (p - 3 + 2) + 1] = 1
                    H0[point_cumulative[0] + counter * (p - 3 + 2) + 1][n] = 1
                    H0[n][point_cumulative[0] + counter * (p - 3 + 2) - 1] = 1
                    H0[point_cumulative[0] + counter * (p - 3 + 2) - 1][n] = 1
                else:
                    H0[n][n + 1] = 1
                    H0[n + 1][n] = 1
                    H0[n][n - 1] = 1
                    H0[n - 1][n] = 1
                    H0[n][point_cumulative[0] + counter*(p-3+2)] = 1
                    H0[point_cumulative[0] + counter*(p-3+2)][n] = 1
                    H0[n][point_cumulative[0] + counter * (p - 3 + 2) + 1] = 1
                    H0[point_cumulative[0] + counter * (p - 3 + 2) + 1][n] = 1
                    H0[n][point_cumulative[0] + counter * (p - 3 + 2) - 1] = 1
                    H0[point_cumulative[0] + counter * (p - 3 + 2) - 1][n] = 1
                    counter += 1

        # if num_levels >= 2:
        #     for n in range(point_cumulative[0], point_cumulative[1]):


    return(H0)


