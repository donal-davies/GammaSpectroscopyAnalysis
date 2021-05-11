import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import math

BACKGROUND_FILENAME = "Data/Background.csv"
DATES = [49,109,141]

def retrieve_data(filename, normalize):
    file = open(filename)

    runtime = 0
    for i in range(22):
        x = file.readline()
        if i == 13:
            runtime = float(x.strip().split(",")[1])

    channel = []
    counts = []

    for line in file:
        sline = line.strip().split(",")

        channel.append(int(sline[0]))
        if normalize:
            counts.append(int(sline[2]) / runtime)
        else:
            counts.append(int(sline[2]))
    file.close()

    channel = np.asarray(channel)
    counts = np.asarray(counts)
    
    return (channel,counts,runtime)

BACKGROUND_DATA = retrieve_data(BACKGROUND_FILENAME,True)

def subtract_background(data):
    return (data[0],data[1]-BACKGROUND_DATA[1],data[2])

def gaussian(x,mu,sigma,A):
    return A*np.exp(-0.5 * ((x-mu)/(sigma))**2)

def fit_gauss(start, end, data):
    p0 = [start + (end-start)/2, (end-start)/4,1]

    firstInd = start
    lastInd = end
    try:
        popt, pcov = curve_fit(gaussian,data[0][start:end],data[1][firstInd:lastInd],p0,sigma=None)
        return (popt,pcov)
    except:
        return (p0,[])

def fit_multiple(start,end,datasets):
    output = []
    for data in datasets:
        p0 = [start + (end-start)/2, (end-start)/4,1]

        try:
            popt, pcov = curve_fit(gaussian,data[0][start:end],data[1][start:end],p0,sigma=None)
            output.append((popt,pcov))
        except:
            output.append((p0,[]))
    return output
        

def sweep_range(point,data):
    fits = [0,0,0]
    for i in range(0,50):
        (popt,pcov) = fit_gauss(max(0,point-i),min(point+i,data[0][-1]),data)
        if popt[1] > fits[1]:
            fits = popt
    return fits

def locate_peaks(data):
    current_max = data[1][-1]
    max_ind = data[0][-1]
    
    for i in data[0][::-1]:
        if data[1][i] > current_max:
            
            current_max = data[1][i]
            max_ind = i


            continue

        if current_max < 1:
            continue

        if np.abs(current_max - data[1][i]) > 0.3*current_max:
            print(max_ind)
            return(sweep_range(max_ind,data))

def plot_data(data):
    plt.plot(data[0],data[1],'b.')
    plt.show()

def fit_linear(x,m,c):
    return m*x + c

def solve_decay(counts,times):
    val1 = (counts[0] + counts[1]) / (counts[1] - counts[2])
    val2 = (-1 * (np.exp(times[1]) - np.exp(times[2]))) / (np.exp(times[0]) + np.exp(times[1]))
    return 0.5 * np.log(val1 * val2)


exps = [15.404638551343046, 3.5122982263832263, 1.897392110522933]
