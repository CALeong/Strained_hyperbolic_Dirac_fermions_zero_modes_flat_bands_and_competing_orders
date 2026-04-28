import numpy as np
from Fundamental.Number_Points import points #Code for getting points per level and total number points

# p = 8 #p value for hyperbolic lattice
# q = 4 #q value for hyperbolic lattice
# num_levels = 4 #Number of levels to consider

def H0(p,q,num_levels,alpha):

    points_perlevel, totalnum = points(p,q,num_levels) #Get points per level and total number of points
    print(points_perlevel)
    print(totalnum)

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

    print(point_cumulative)

    #Initialize H0 matrix (original coordinates)
    H0 = np.zeros((int(totalnum),int(totalnum)),dtype=np.complex_)
    t = 1
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
            x1 = alpha/p
            tn1 = t*np.exp(1j*x1)

            for n in range(point_cumulative[0]): #First Level
                if n == 0: # First point on level
                    H0[n][n+1] = tn1 #Connected to point next to it
                    H0[n+1][n] = np.conj(tn1)
                    if num_levels != 1:
                        H0[n][point_cumulative[0]] = t
                        H0[point_cumulative[0]][n] = t
                elif n == point_cumulative[0]-1: #Last point on level
                    H0[n][0] = tn1 #Connected to first point in level
                    H0[0][n] = np.conj(tn1)
                    #n * (p-q) is because each point connected to level one point is separated by (p-q) points
                    #Thus point_cumulative[0] + n * (p - q) gives value for the point connected to nth first level point
                    if num_levels != 1:
                        H0[n][point_cumulative[0] + n * (p - q)] = t #Code for finding connections with next level
                        H0[point_cumulative[0] + n * (p - q)][n] = t
                else: #All other points
                    H0[n][n + 1] = tn1 #Connected to next point on first level
                    H0[n + 1][n] = np.conj(tn1)
                    H0[n][n - 1] = np.conj(tn1) #Connected to previous point on first level
                    H0[n - 1][n] = tn1
                    if num_levels != 1:
                        H0[n][point_cumulative[0]+ n*(p-q)] = t #Same deal as above with connecting to points on next level
                        H0[point_cumulative[0]+ n*(p-q)][n] = t
        threecounter,fourcounter,counter,subcounter, onsub = reinitialize_counter()
        if num_levels >= 2:
            x2 = (alpha+x1)/(p-3)
            tn2 = t*np.exp(1j*x2)
            for n in range(point_cumulative[0],point_cumulative[1]): #Second Level
                if n == point_cumulative[0]: #First point
                    H0[n][n + 1] = tn2 #Point next to it is connected
                    H0[n + 1][n] = np.conj(tn2) #Dont worry about previous point as that is handled in "Last Point" code
                    counter += 1 #Add one to counter to start counting (relevant in later code)
                elif n == point_cumulative[1]-1: #Last Point
                    H0[n][point_cumulative[0]] = tn2 #Connect last point with first point on second level
                    H0[point_cumulative[0]][n] = np.conj(tn2)
                    #Fourcounter and three counter explained later in code
                    #Connect last point of level with points on next level
                    if num_levels != 2:
                        H0[n][point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)] = t
                        H0[point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)][n] = t
                elif counter == 0: #Points where no connected points on next level
                    counter += 1
                    H0[n][n + 1] = tn2 #Connect with next point on level
                    H0[n + 1][n] = np.conj(tn2)
                    H0[n][n - 1] = np.conj(tn2) #Connect with previous point on level
                    H0[n - 1][n] = tn2
                    fourcounter += 1 #Add fourcounter since counter==0 point occurs at four-border polygon
                elif counter != (p-q-1): #Points where there is a connected point on next level
                    counter += 1
                    H0[n][n + 1] = tn2 #Connect with next point on level
                    H0[n + 1][n] = np.conj(tn2)
                    H0[n][n - 1] = np.conj(tn2) #Connect with previous point on level
                    H0[n - 1][n] = tn2
                    #fourcounter = number of four border polygons passed
                    #threecounter = number of three border polygons passed
                    #n-border polygon means polygon with n sides that do not have next level points on both ends of edge
                    #(p-q-1) next level points on four border polygon
                    #(p-q) next level points on three border polygon
                    #Calculation thus gives point on next level connected to point on second level
                    if num_levels != 2:
                        H0[n][point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)] = t
                        H0[point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)][n] = t
                        threecounter += 1 #Add threecounter since each point traverses a three border polygon unless counter==0
                elif counter == (p-q-1): #Last point that is connected with next level before counter==0 point occurs
                    counter = 0 #set counter=0 so code knows we are at four border polygon point
                    H0[n][n + 1] = tn2 #Connect point with next point on label
                    H0[n + 1][n] = np.conj(tn2)
                    H0[n][n - 1] = np.conj(tn2) #Connect point with last point on label
                    H0[n - 1][n] = tn2
                    #Same deal as before with getting point on next level connected to second level point
                    if num_levels != 2:
                        H0[n][point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)] = t
                        H0[point_cumulative[1] + (p - q - 1) * fourcounter + (p - q) * (threecounter)][n] = t

        threecounter,fourcounter,counter,subcounter, onsub = reinitialize_counter()
        onsubcounter = 0
        if num_levels >= 3:
            x3 = (alpha+x2)/(p-3)
            x4 = (alpha+2*x2)/(p-4)
            tn3 = t*np.exp(1j*x3)
            tn4 = t*np.exp(1j*x4)
            for n in range(point_cumulative[1],point_cumulative[2]): #Third Level
                if n == point_cumulative[1]: #First point on third level
                    H0[n][n + 1] = tn3 #Connect point with next point on third level
                    H0[n + 1][n] = np.conj(tn3)
                    counter += 1 #Start counter for later
                    H0[n][point_cumulative[2]-1] = np.conj(tn4)
                    H0[point_cumulative[2]-1][n] = tn4
                elif n == point_cumulative[2]-1:
                    threecounter += 1
                    H0[n][point_cumulative[1]] = tn4
                    H0[point_cumulative[1]][n] = np.conj(tn4)
                    if num_levels != 3:
                        H0[n][point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)] = t
                        H0[point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)][n] = t
                elif subcounter == (p-q-2):
                    onsubcounter += 1
                    if onsubcounter == 1:
                        H0[n][n + 1] = tn4
                        H0[n + 1][n] = np.conj(tn4)
                        if num_levels != 3:
                            H0[n][point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)] = t
                            H0[point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)][n] = t
                    elif onsubcounter == (p-q-1):
                        H0[n][n + 1] = tn3
                        H0[n + 1][n] = np.conj(tn3)
                        counter = 1
                        onsubcounter = 0
                        subcounter = 0
                        fourcounter += 1
                    else:
                        threecounter += 1
                        H0[n][n + 1] = tn4
                        H0[n + 1][n] = np.conj(tn4)
                        if num_levels != 3:
                            H0[n][point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)] = t
                            H0[point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)][n] = t
                elif counter == 1:
                    H0[n][n + 1] = tn3
                    H0[n + 1][n] = np.conj(tn3)
                    if num_levels != 3:
                        H0[n][point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)] = t
                        H0[point_cumulative[2] + threecounter * (p - q) + fourcounter * (p - q - 1)][n] = t
                    counter += 1
                elif counter != (p-q-1) and counter != 0:
                    threecounter += 1
                    H0[n][n+1] = tn3
                    H0[n+1][n] = np.conj(tn3)
                    if num_levels != 3:
                        H0[n][point_cumulative[2]+threecounter*(p-q)+fourcounter*(p-q-1)] = t
                        H0[point_cumulative[2]+threecounter*(p-q)+fourcounter*(p-q-1)][n] = t
                    counter += 1
                elif counter == (p-q-1):
                    counter = 0
                    threecounter += 1
                    H0[n][n + 1] = tn3
                    H0[n + 1][n] = np.conj(tn3)
                    if num_levels != 3:
                        H0[n][point_cumulative[2] + threecounter * (p - q) + fourcounter*(p-q-1)] = t
                        H0[point_cumulative[2] + threecounter * (p - q) + fourcounter*(p-q-1)][n] = t
                elif counter == 0:
                    counter += 1
                    subcounter += 1
                    if subcounter != (p-q-2):
                        H0[n][n + 1] = tn3
                        H0[n + 1][n] = np.conj(tn3)
                    elif subcounter == (p-q-2):
                        H0[n][n + 1] = tn4
                        H0[n + 1][n] = np.conj(tn4)
                    fourcounter += 1
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
            x5 = (alpha+x3)/(p-3)
            x6 = (alpha+x4)/(p-3)
            x7 = (alpha+2*x3)/(p-4)
            x8 = (alpha+x3+x4)/(p-4)
            tn5 = t*np.exp(1j*x5)
            tn6 = t*np.exp(1j*x6)
            tn7 = t*np.exp(1j*x7)
            tn8 = t*np.exp(1j*x8)
            for n in range(point_cumulative[2],point_cumulative[3]): #Fourth Level
                if n == point_cumulative[2]: #First point
                    H0[n+1][n] = np.conj(tn5)
                    H0[n][n+1] = tn5
                    H0[n][point_cumulative[3]-1] = np.conj(tn8)
                    H0[point_cumulative[3]-1][n] = tn8
                    counter += 1
                elif n == point_cumulative[3]-1: #Last point
                    H0[n - 1][n] = tn8
                    H0[n][n - 1] = np.conj(tn8)
                    threecounter += 1
                    if num_levels > 4:
                        H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = t
                        H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = t
                elif motif2 == True: #On part of lattice that is motif2 (described above)
                    if subcounter == (p-q-3): #Means now on subcounter portion at end of motif2
                        onsubcounter += 1 #counter to bring code through subcounter
                        if onsubcounter == 1:
                            H0[n][n+1] = tn8
                            H0[n+1][n] = np.conj(tn8)
                            H0[n][n - 1] = np.conj(tn8)
                            H0[n - 1][n] = tn8
                            if num_levels > 4:
                                H0[n][point_cumulative[3]+fourcount*fourcounter+threecount*threecounter] = t
                                H0[point_cumulative[3]+fourcount*fourcounter+threecount*threecounter][n] = t
                        elif onsubcounter != (p-q-1):
                            H0[n][n + 1] = tn8
                            H0[n + 1][n] = np.conj(tn8)
                            H0[n][n - 1] = np.conj(tn8)
                            H0[n - 1][n] = tn8
                            threecounter += 1
                            if num_levels > 4:
                                H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = t
                                H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = t
                        elif onsubcounter == (p-q-1): #Last point in subcounter, reset values so they are ready for next point
                            fourcounter += 1
                            onsubcounter = 0
                            subcounter = 0
                            subsubcounter = 0
                            counter = 1
                            H0[n][n + 1] = tn5
                            H0[n + 1][n] = np.conj(tn5)
                            H0[n][n - 1] = np.conj(tn8)
                            H0[n - 1][n] = tn8
                            motif1 = True #Switch back to motif1
                            motif2 = False
                    elif counter == 1: #Cycle through counter portions
                        H0[n][n + 1] = tn6
                        H0[n + 1][n] = np.conj(tn6)
                        H0[n][n - 1] = np.conj(tn6)
                        H0[n - 1][n] = tn6
                        if num_levels > 4:
                            H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = t
                            H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = t
                        counter += 1
                    elif counter != (p-q):
                        H0[n][n + 1] = tn6
                        H0[n + 1][n] = np.conj(tn6)
                        H0[n][n - 1] = np.conj(tn6)
                        H0[n - 1][n] = tn6
                        threecounter += 1
                        if num_levels > 4:
                            H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = t
                            H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = t
                        counter += 1
                    elif counter == (p-q) and subcounter != (p-q-3)-1:
                        H0[n][n + 1] = tn6
                        H0[n + 1][n] = np.conj(tn6)
                        H0[n][n - 1] = np.conj(tn6)
                        H0[n - 1][n] = tn6
                        fourcounter += 1
                        subcounter += 1 #Add to subcounter. Once threshold reached, tells code that it is on subcounter portion
                        counter = 1
                    elif counter == (p-q) and subcounter == (p-q-3)-1:
                        H0[n][n + 1] = tn8
                        H0[n + 1][n] = np.conj(tn8)
                        H0[n][n - 1] = np.conj(tn6)
                        H0[n - 1][n] = tn6
                        fourcounter += 1
                        subcounter += 1 #Add to subcounter. Once threshold reached, tells code that it is on subcounter portion
                        counter = 1
                elif motif1 == True: #motif1 protion of lattice described above
                    if subcounter == (p-q-2): #subcounter portion of motif1
                        onsubcounter += 1 #cycle through subcounter portion
                        if onsubcounter == 1 and subsubcounter != (p-q-2)-1:
                            H0[n][n+1] = tn7
                            H0[n+1][n] = np.conj(tn7)
                            H0[n][n - 1] = np.conj(tn7)
                            H0[n - 1][n] = tn7
                            if num_levels > 4:
                                H0[n][point_cumulative[3]+fourcount*fourcounter+threecount*threecounter] = t
                                H0[point_cumulative[3]+fourcount*fourcounter+threecount*threecounter][n] = t
                        elif onsubcounter == 1 and subsubcounter == (p-q-2)-1:
                            H0[n][n+1] = tn8
                            H0[n+1][n] = np.conj(tn8)
                            H0[n][n - 1] = np.conj(tn8)
                            H0[n - 1][n] = tn8
                            if num_levels > 4:
                                H0[n][point_cumulative[3]+fourcount*fourcounter+threecount*threecounter] = t
                                H0[point_cumulative[3]+fourcount*fourcounter+threecount*threecounter][n] = t
                        elif onsubcounter != (p-q-1) and subsubcounter != (p-q-2)-1:
                            H0[n][n + 1] = tn7
                            H0[n + 1][n] = np.conj(tn7)
                            H0[n][n - 1] = np.conj(tn7)
                            H0[n - 1][n] = tn7
                            threecounter += 1
                            if num_levels > 4:
                                H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = t
                                H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = t
                        elif onsubcounter != (p-q-1) and subsubcounter == (p-q-2)-1:
                            H0[n][n + 1] = tn8
                            H0[n + 1][n] = np.conj(tn8)
                            H0[n][n - 1] = np.conj(tn8)
                            H0[n - 1][n] = tn8
                            threecounter += 1
                            if num_levels > 4:
                                H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = t
                                H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = t
                        elif onsubcounter == (p-q-1) and subsubcounter != (p-q-2)-1: #Last point, reset values for next point
                            fourcounter += 1
                            onsubcounter = 0
                            subcounter = 0
                            counter = 1
                            H0[n][n + 1] = tn5
                            H0[n + 1][n] = np.conj(tn5)
                            H0[n][n - 1] = np.conj(tn7)
                            H0[n - 1][n] = tn7
                            subsubcounter += 1
                            if subsubcounter == (p-q-2): #If certain number of subcounters passed on motif1, need to switch to motif2
                                motif1 = False
                                motif2 = True
                        elif onsubcounter == (p-q-1) and subsubcounter == (p-q-2)-1: #Last point, reset values for next point
                            fourcounter += 1
                            onsubcounter = 0
                            subcounter = 0
                            counter = 1
                            H0[n][n + 1] = tn6
                            H0[n + 1][n] = np.conj(tn6)
                            H0[n][n - 1] = np.conj(tn8)
                            H0[n - 1][n] = tn8
                            subsubcounter += 1
                            if subsubcounter == (p-q-2): #If certain number of subcounters passed on motif1, need to switch to motif2
                                motif1 = False
                                motif2 = True
                    elif counter == 1: #cycle through counter portion of motif1 (with breaks to subcounter when at those points)
                        H0[n][n + 1] = tn5
                        H0[n + 1][n] = np.conj(tn5)
                        H0[n][n - 1] = np.conj(tn5)
                        H0[n - 1][n] = tn5
                        if num_levels > 4:
                            H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = t
                            H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = t
                        counter += 1
                    elif counter != (p-q):
                        H0[n][n + 1] = tn5
                        H0[n + 1][n] = np.conj(tn5)
                        H0[n][n - 1] = np.conj(tn5)
                        H0[n - 1][n] = tn5
                        threecounter += 1
                        if num_levels > 4:
                            H0[n][point_cumulative[3] + fourcount * fourcounter + threecount * threecounter] = t
                            H0[point_cumulative[3] + fourcount * fourcounter + threecount * threecounter][n] = t
                        counter += 1
                    elif counter == (p-q):
                        subcounter += 1 #counter to keep track so code knows when its on subcounter of motif1
                        if subcounter == (p-q-2) and subsubcounter != (p-q-2)-1:
                            H0[n][n + 1] = tn7
                            H0[n + 1][n] = np.conj(tn7)
                        elif subcounter == (p-q-2) and subsubcounter == (p-q-2)-1:
                            H0[n][n + 1] = tn8
                            H0[n + 1][n] = np.conj(tn8)
                        else:
                            H0[n][n + 1] = tn5
                            H0[n + 1][n] = np.conj(tn5)
                        H0[n][n - 1] = np.conj(tn5)
                        H0[n - 1][n] = tn5
                        fourcounter += 1
                        counter = 1

        if num_levels == 5:
            x9 = (alpha+x5)/(p-3)
            x10 = (alpha+x7)/(p-3)
            x11 = (alpha+2*x5)/(p-4)
            x12 = (alpha + x5 + x7)/(p-4)
            x13 = (alpha +x6 + x8)/(p-4)
            x14 = (alpha + x8)/(p-3)
            x15 = (alpha + x6)/(p-3)
            x16 = (alpha + x5 + x8)/(p-4)
            x17 = (alpha + 2*x6)/(p-4)
            tn9 = t*np.exp(1j*x9)
            tn10 = t*np.exp(1j*x10)
            tn11 = t*np.exp(1j*x11)
            tn12 = t*np.exp(1j*x12)
            tn13 = t*np.exp(1j*x13)
            tn14 = t*np.exp(1j*x14)
            tn15 = t*np.exp(1j*x15)
            tn16 = t*np.exp(1j*x16)
            tn17 = t*np.exp(1j*x17)

            x9motif = np.repeat(tn9,(p-q)*(p-q-2))
            x10motif = np.repeat(tn10,(p-q)*(p-q-3))
            x11motif = np.repeat(tn11,(p-q-1))
            x12motif = np.repeat(tn12,(p-q-1))
            x13motif = np.repeat(tn13,(p-q-1))
            x14motif = np.repeat(tn14,(p-q)*(p-q-3))
            x15motif = np.repeat(tn15,(p-q)*(p-q-2))
            x16motif = np.repeat(tn16,(p-q-1))
            x17motif = np.repeat(tn17,(p-q-1))

            x9x11pattern = np.tile(np.concatenate((x9motif, x11motif),axis=None),(p-q-3))
            x12x10pattern = np.concatenate((x12motif,x10motif,x12motif),axis=None)
            x9x11x12x10_totalpattern = np.tile(np.concatenate((x9x11pattern,x9motif,x12x10pattern),axis=None),(p-q-3))
            x15x17pattern = np.tile(np.concatenate((x15motif,x17motif),axis=None),(p-q-4))

            # motif_sequence = np.concatenate((x9motif,x11motif,x9motif,x11motif,x9motif,x12motif,x10motif,x12motif,x9motif,x11motif,x9motif,x11motif,x9motif,x12motif,x10motif,x12motif,x9motif,x11motif,x9motif,x11motif,x9motif,x16motif,x14motif,x13motif,x15motif,x17motif,x15motif,x13motif,x14motif,x16motif),axis=None)
            if (p-q-4) < 0:
                print('Currently not working')
                # motif_sequence = np.concatenate((np.repeat(np.array([np.repeat(np.array([x9motif,x11motif]),(p-q-3),axis=None),x9motif,x12motif,x10motif,x12motif]),(p-q-3),axis=None),np.array([np.repeat(np.array([x9motif,x11motif]),(p-q-3),axis=None),x9motif,x16motif,x14motif,x13motif]),np.repeat(np.array([x15motif,x17motif]),(p-q-4),axis=None),x15motif,x13motif,x16motif),axis=None)
            else:
                # motif_sequence = np.concatenate((np.repeat(np.array([np.repeat(np.array([x9motif,x11motif]),(p-q-3),axis=None),x9motif,x12motif,x10motif,x12motif]),(p-q-3),axis=None),np.array([np.repeat(np.array([x9motif,x11motif]),(p-q-3),axis=None),x9motif,x16motif,x14motif,x13motif]),np.repeat(np.array([x15motif,x13motif]),(p-q-3),axis=None),x14motif,x16motif),axis=None)
                # motif_sequence = np.concatenate((x9x11pattern,x12x10pattern,x9x11pattern,x12x10pattern,x9x11pattern,x16motif,x14motif,x13motif,x15motif,x17motif,x15motif,x13motif,x14motif,x16motif),axis=None,dtype=np.complex_)
                motif_sequence = np.concatenate((x9x11x12x10_totalpattern,x9x11pattern,x9motif,x16motif,x14motif,x13motif,x15x17pattern,x15motif,x13motif,x14motif,x16motif),axis=None,dtype=np.complex_)

            motif_i = 0
            m = 0
            for n in range(point_cumulative[3],point_cumulative[4]):
                # print(motif_i)
                # print(n, n-point_cumulative[3]-m*np.size(motif_sequence,0),np.size(motif_sequence,0))
                # print(n-point_cumulative[3]-m*np.size(motif_sequence,0) == np.size(motif_sequence,0)-1)
                if n == point_cumulative[4]-1:
                    # motif_i = 0
                    H0[n][point_cumulative[3]] = motif_sequence[motif_i]
                    H0[point_cumulative[3]][n] = np.conj(motif_sequence[motif_i])
                elif n-point_cumulative[3]-m*np.size(motif_sequence,0) == np.size(motif_sequence,0)-1:
                    m += 1
                    H0[n][n + 1] = motif_sequence[motif_i]
                    H0[n + 1][n] = np.conj(motif_sequence[motif_i])
                    motif_i = 0
                elif n-point_cumulative[3]-m*np.size(motif_sequence,0) < np.size(motif_sequence,0):
                    H0[n][n+1] = motif_sequence[motif_i]
                    H0[n+1][n] = np.conj(motif_sequence[motif_i])
                    motif_i += 1

            print(np.size(motif_sequence,0))
    ###q=4 is NOT ready. Still need to code it. What is there is just copied and pasted
    ###from Hamiltoniana H0 q=4 code

    if q == 4:  # q=4 H0 populating code

        x1 = alpha/p
        x2 = (alpha+x1)/(p-3)
        x3 = alpha/(p-2)
        x4 = (alpha+x2)/(p-3)
        x5 = (alpha+x3)/(p-3)
        x6 = (alpha+x4)/(p-3)
        tn1 = t*np.exp(1j*x1)
        tn2 = t*np.exp(1j*x2)
        tn3 = t*np.exp(1j*x3)
        tn4 = t*np.exp(1j*x4)
        tn5 = t*np.exp(1j*x5)
        tn6 = t*np.exp(1j*x6)

        threecounter, fourcounter, counter, subcounter, onsub = reinitialize_counter()
        oddcounter = 0
        oddcount = (p - q + 1)
        evencounter = 0
        evencount = 1
        for n in range(point_cumulative[0]):  # First level
            if n == 0:
                H0[n][n + 1] = tn1
                H0[n + 1][n] = np.conj(tn1)
                H0[n][point_cumulative[0] - 1] = np.conj(tn1)
                H0[point_cumulative[0] - 1][n] = tn1
                H0[n][point_cumulative[0]] = t
                H0[point_cumulative[0]][n] = t
                H0[n][point_cumulative[1] - 1] = t
                H0[point_cumulative[1] - 1][n] = t
                counter += 1
            elif n == point_cumulative[0] - 1:
                evencounter += 1
                oddcounter += 1
                H0[n][n - 1] = np.conj(tn1)
                H0[n - 1][n] = tn1
                H0[n][point_cumulative[0] + oddcounter * oddcount + evencounter * evencount] = t
                H0[point_cumulative[0] + oddcounter * oddcount + evencounter * evencount][n] = t
                H0[point_cumulative[0] + oddcounter * oddcount + evencounter * evencount - 1][n] = t
                H0[n][point_cumulative[0] + oddcounter * oddcount + evencounter * evencount - 1] = t
            else:
                evencounter += 1
                oddcounter += 1
                H0[n][n + 1] = tn1
                H0[n + 1][n] = np.conj(tn1)
                H0[n][n - 1] = np.conj(tn1)
                H0[n - 1][n] = tn1
                H0[n][point_cumulative[0] + oddcounter * oddcount + evencounter * evencount] = t
                H0[point_cumulative[0] + oddcounter * oddcount + evencounter * evencount][n] = t
                H0[n][point_cumulative[0] + oddcounter * oddcount + evencounter * evencount - 1] = t
                H0[point_cumulative[0] + oddcounter * oddcount + evencounter * evencount - 1][n] = t

        threecounter, fourcounter, counter, subcounter, onsub = reinitialize_counter()
        oddcounter = 0
        oddcount = (p - q + 1)
        evencounter = 0
        evencount = 1
        if num_levels > 1:
            for n in range(point_cumulative[0], point_cumulative[1]):
                if n == point_cumulative[0]:
                    H0[n][n + 1] = tn2
                    H0[n + 1][n] = np.conj(tn2)
                    if num_levels > 2:
                        H0[n][point_cumulative[2] - 1] = np.conj(tn3)
                        H0[point_cumulative[2] - 1][n] = tn3
                        H0[n][point_cumulative[1]] = t
                        H0[point_cumulative[1]][n] = t
                    counter += 1
                elif counter == 0:
                    H0[n][n + 1] = tn2
                    H0[n + 1][n] = np.conj(tn2)
                    if num_levels > 2:
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount] = t
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount][n] = t
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1] = np.conj(tn3)
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1][n] = tn3
                    counter += 1
                elif counter == (p - q + 1):
                    oddcounter += 1
                    evencounter += 1
                    H0[n][n - 1] = np.conj(tn2)
                    H0[n - 1][n] = tn2
                    if num_levels > 2:
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount] = tn3
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount][n] = np.conj(tn3)
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1] = t
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1][n] = t
                    counter = 0
                    oddcounter += 1
                elif counter != (p - q + 1):
                    oddcounter += 1
                    evencounter += 1
                    H0[n][n + 1] = tn2
                    H0[n + 1][n] = np.conj(tn2)
                    H0[n][n - 1] = np.conj(tn2)
                    H0[n - 1][n] = tn2
                    if num_levels > 2:
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount] = t
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount][n] = t
                        H0[n][point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1] = t
                        H0[point_cumulative[1] + oddcounter * oddcount + evencounter * evencount - 1][n] = t
                    counter += 1
        threecounter, fourcounter, counter, subcounter, onsub = reinitialize_counter()
        oddcounter = 0
        oddcount = (p - q + 1)
        evencounter = 0
        evencount = 1
        subsubcounter = 0
        onsubcounter = 0
        if num_levels > 2:
            for n in range(point_cumulative[1], point_cumulative[2]):  # Third Level
                if n == point_cumulative[1]:
                    H0[n][n + 1] = tn4
                    H0[n + 1][n] = np.conj(tn4)
                    if num_levels > 3:
                        H0[n][point_cumulative[2]] = t
                        H0[point_cumulative[2]][n] = t
                        H0[n][point_cumulative[3] - 1] = np.conj(tn5)
                        H0[point_cumulative[3] - 1][n] = tn5
                    counter += 1
                elif subcounter == (p - q + 1):
                    if onsubcounter == 0:
                        oddcounter += 1
                        H0[n][n + 1] = tn3
                        H0[n + 1][n] = np.conj(tn3)
                        if num_levels > 3:
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = t
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = t
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = t
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = t
                        onsubcounter += 1
                    elif onsubcounter == (p - q + 1):
                        onsubcounter = 0
                        subcounter = 0
                        counter = 1
                        oddcounter += 1
                        H0[n][n + 1] = tn4
                        H0[n + 1][n] = np.conj(tn4)
                        if num_levels > 3:
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = t
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = t
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = np.conj(tn5)
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = tn5
                    elif onsubcounter == (p - q):
                        oddcounter += 1
                        evencounter += 1
                        H0[n][n - 1] = np.conj(tn3)
                        H0[n - 1][n] = tn3
                        if num_levels > 3:
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = t
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = t
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = t
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = t
                        onsubcounter += 1
                    elif onsubcounter != (p - q + 1):
                        oddcounter += 1
                        evencounter += 1
                        H0[n][n + 1] = tn3
                        H0[n + 1][n] = np.conj(tn3)
                        H0[n][n - 1] = np.conj(tn3)
                        H0[n - 1][n] = tn3
                        if num_levels > 3:
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = t
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = t
                            H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = t
                            H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = t
                        onsubcounter += 1
                elif counter == 0:
                    oddcounter += 1
                    H0[n][n + 1] = tn4
                    H0[n + 1][n] = np.conj(tn4)
                    if num_levels > 3:
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = t
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = t
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = np.conj(tn3)
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = tn3
                    counter += 1
                elif counter != (p - q + 1):
                    oddcounter += 1
                    evencounter += 1
                    H0[n][n + 1] = tn4
                    H0[n + 1][n] = np.conj(tn4)
                    H0[n][n - 1] = np.conj(tn4)
                    H0[n - 1][n] = tn4
                    if num_levels > 3:
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = t
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = t
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = t
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = t
                    counter += 1
                elif counter == (p - q + 1):
                    subcounter += 1
                    oddcounter += 1
                    evencounter += 1
                    H0[n][n - 1] = np.conj(tn4)
                    H0[n - 1][n] = tn4
                    if num_levels > 3:
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount] = tn3
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount][n] = np.conj(tn3)
                        H0[n][point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1] = t
                        H0[point_cumulative[2] + evencounter * evencount + oddcounter * oddcount - 1][n] = t
                    counter = 0

        threecounter, fourcounter, counter, subcounter, onsub = reinitialize_counter()
        oddcounter = 0
        oddcount = (p - q + 1)
        evencounter = 0
        evencount = 1
        subsubcounter = 0
        onsubcounter = 0
        if num_levels > 3:
            for n in range(point_cumulative[2], point_cumulative[3]):  # Fourth Level
                if n == point_cumulative[2]:
                    H0[n][n + 1] = tn6
                    H0[n + 1][n] = np.conj(tn6)
                    counter += 1
                elif n == point_cumulative[3] - 1:
                    H0[n][n - 1] = np.conj(tn5)
                    H0[n - 1][n] = tn5
                elif subsubcounter == (p - q + 1) and subcounter == (p - q):
                    H0[n][n + 1] = tn5
                    H0[n + 1][n] = np.conj(tn5)
                    counter = 0
                    subcounter = (p - q + 1)  # Push it to subcounter routine
                    onsubcounter = 1
                elif subcounter == (p - q + 1):
                    if onsubcounter == 0:
                        H0[n][n + 1] = tn3
                        H0[n + 1][n] = np.conj(tn3)
                        onsubcounter += 1
                    elif onsubcounter == (p - q):
                        H0[n][n - 1] = np.conj(tn3)
                        H0[n - 1][n] = tn3
                        onsubcounter = 0
                        counter = 0
                        subcounter = 0
                        subsubcounter += 1
                        if subsubcounter == (p - q + 1) + 1:  # Check if subsub portion passed and if it did then reset subsubcounter
                            subsubcounter = 0
                    elif onsubcounter != (p - q):
                        H0[n][n + 1] = tn3
                        H0[n + 1][n] = np.conj(tn3)
                        H0[n][n - 1] = np.conj(tn3)
                        H0[n - 1][n] = tn3
                        onsubcounter += 1
                elif counter == (p - q + 1):
                    H0[n][n - 1] = np.conj(tn6)
                    H0[n - 1][n] = tn6
                    counter = 0
                    subcounter += 1
                elif counter == 0:
                    H0[n][n + 1] = tn6
                    H0[n + 1][n] = np.conj(tn6)
                    counter += 1
                elif counter != (p - q + 1) and counter != 0:
                    H0[n][n - 1] = np.conj(tn6)
                    H0[n - 1][n] = tn6
                    H0[n][n + 1] = tn6
                    H0[n + 1][n] = np.conj(tn6)
                    counter += 1

    '''
    if q == 4: #q=4 H0 populating code
        if num_levels >= 1:
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
                    if num_levels > 1:
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
                    if num_levels > 1:
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
                    if num_levels > 1:
                        H0[n][point_cumulative[0]+oddcounter*oddcount+evencounter*evencount] = 1
                        H0[point_cumulative[0]+oddcounter*oddcount+evencounter*evencount][n] = 1
                        H0[n][point_cumulative[0]+oddcounter*oddcount+evencounter*evencount-1] = 1
                        H0[point_cumulative[0]+oddcounter*oddcount+evencounter*evencount-1][n] = 1

        if num_levels >= 2:
            threecounter, fourcounter, counter, subcounter, onsub = reinitialize_counter()
            oddcounter = 0
            oddcount = (p - q + 1)
            evencounter = 0
            evencount = 1
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

        if num_levels >= 3:
            threecounter, fourcounter, counter, subcounter, onsub = reinitialize_counter()
            oddcounter = 0
            oddcount = (p - q + 1)
            evencounter = 0
            evencount = 1
            subsubcounter = 0
            onsubcounter = 0
            for n in range(point_cumulative[1],point_cumulative[2]): #Third Level
                if n == point_cumulative[1]:
                    if num_levels > 3:
                        H0[n][point_cumulative[2]] = 1
                        H0[point_cumulative[2]][n] = 1
                        H0[n][point_cumulative[3]-1] = 1
                        H0[point_cumulative[3]-1][n] = 1
                    H0[n][n+1] = 1
                    H0[n+1][n] = 1
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

        if num_levels == 4:
            threecounter, fourcounter, counter, subcounter, onsub = reinitialize_counter()
            oddcounter = 0
            oddcount = (p - q + 1)
            evencounter = 0
            evencount = 1
            subsubcounter = 0
            onsubcounter = 0
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
    '''
    return(H0)

def Number_Plaquets(p,q,num_levels):
    plaquets_per_level = np.array([])
    number_plaquets = 0
    if q == 3:
        plaquets_per_threeside = (p-4)
        plaquets_per_fourside = (p-5)
        if num_levels >= 1:
            number_plaquets = number_plaquets + 1
            plaquets_per_level = np.append(plaquets_per_level, 1)
        if num_levels >= 2:
            number_plaquets = number_plaquets + p
            plaquets_per_level = np.append(plaquets_per_level, p)
        if num_levels >= 3:
            num_prev_threesides = p
            num_prev_foursides = 0
            for n in range(3,num_levels+1):
                addterm = num_prev_threesides*plaquets_per_threeside + num_prev_foursides*plaquets_per_fourside
                number_plaquets = number_plaquets + addterm
                plaquets_per_level = np.append(plaquets_per_level, addterm)
                num_prev_threesides = (p-5)*num_prev_threesides + (p-6)*num_prev_foursides
                num_prev_foursides = plaquets_per_level[n-2]
    return(plaquets_per_level, number_plaquets)

