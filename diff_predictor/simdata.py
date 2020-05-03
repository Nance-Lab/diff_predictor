import numpy as np
import pandas as pd
import boto3
import diff_classifier.aws as aws
import os
import math
import matplotlib.pyplot as plt

from io import StringIO
from csv import writer

import secrets
import diff_classifier.features as ft

def directed_diffusion(n_particles=1000, size=1600, time_max=50, vm=None, diff=None, df=None):

    # Directed Diffusion
    # space_units == 'pixels'
    # time_units == 'frames'

    if type(df) == type(None) or (type(df) == type(pd.DataFrame()) and df.empty):
        df = pd.DataFrame(columns=['Frame', 'Track_ID', 'X', 'Y', 'Type'])
        append_track = 0
    else:
        append_track = df['Track_ID'].max()+1

    if diff == None:
        diff = np.random.rand()*5  # diffusion coeff, pixels^2/frame
    if vm == None:
        # mean velocity, pixel/frame
        vm = np.random.normal(loc=2.04, scale=1.25)

    # Time step between acquisition; fast acquisition!
    dt = 1  # Frame

    # Creating positional array
    pos_out = [[] for x in range(n_particles*651+2)]
    pos_out[0] = ['Frame', 'Track_ID', 'X', 'Y', 'Type']  # First row is names
    m = 2

    k = np.sqrt(2*diff*dt)

    for i in range(n_particles):
        brownian_inf = np.random.rand()
        n_time_steps = np.random.randint(5, time_max)
        # Setting time variable
        time = np.array(range(n_time_steps-1))*dt
        # Setting velocity orientation, theta
        theta = 2*np.pi*np.random.uniform()
        # Mean velocity
        v = vm*(1+0.25*np.random.randn())
        # Initial position
        x0 = size*np.random.uniform(size=(1, 2))
        # Instantaneous displacement
        dx_brownian = k**brownian_inf*np.random.randn(n_time_steps, 2)
        dx_directed = v*dt * \
            np.array([np.cos(theta)*np.ones(n_time_steps),
                      np.sin(theta)*np.ones(n_time_steps)])
        # Integrate uncorrelated displacement
        dx = dx_brownian + np.transpose(dx_directed)
        dx[0] = x0
        x = np.cumsum(dx, 0)
        for j in range(2, 653):
            if j-2 >= n_time_steps:
                pos_out[m] =  np.array([j-2, i+append_track, np.nan, np.nan, 'directed'])
                m += 1
                continue
            pos_out[m] =  np.array([j-2, i+append_track,
                          x[j-2][0], x[j-2][1], 'directed'])
            m += 1
    output = StringIO()
    csv_writer = writer(output)
    for row in pos_out:
#         print(row)
        csv_writer.writerow(row)
    output.seek(0)  # we need to get back to the start of the StringIO
    df2 = pd.read_csv(output)
    return pd.concat([df, df2])

def anomalous_diffusion(n_particles=1000, size=1600, time_max=50, alpha=None, mu=None, diff=None, df=None):
    # Anomalous Diffusion (fractional brownian motion) using weiner process
    # space_units == 'pixels'
    # time_units == 'frames'

    def fbm_gauss(n_time_steps, alpha):
        """
        The following function uses the circulant embedding method to
        calculate the Wiener fractional Brownian motion distribution. This is
        described by Volker Schmidt in "Stochastic Geometry, Spatial
        Statistics and Random Fields"

        Parameters
        ----------
        n_time_steps : integer
            Length of time series to simulate
        alpha : float
            Desired alpha variable s.t. alpha E(0,2)

        Returns
        -------
        w : np.array
            Wiener distribution {W_t, t>=0} of fBm. Cumsum of the fractioanl
            Gaussian noise
        """
        r = np.zeros(shape=(n_time_steps+1, 1))
        r[0] = 1
        h = alpha/2
        for k in range(1, n_time_steps+1):
            r[k] = 0.5*((k+1)**(alpha) - 2*k**(alpha) + (k-1)**(alpha))
        r = np.concatenate((r, np.flip(r[1:-1])))
        lamb = np.real(np.fft.fft(r, axis=0))/(2*(n_time_steps))
        w = np.fft.fft(np.sqrt(lamb) * np.random.randn(2*n_time_steps,
                                                       1) + 1j * np.random.randn(2*n_time_steps, 1), axis=0)
        w = n_time_steps**(-h)*np.cumsum(np.real(w[0:n_time_steps+1]))
        return w

    if type(df) == type(None) or (type(df) == type(pd.DataFrame()) and df.empty):
        df = pd.DataFrame(columns=['Frame', 'Track_ID', 'X', 'Y', 'Type'])
        append_track = 0
    else:
        append_track = df['Track_ID'].max()+1
    if mu == None:
        mu = np.random.rand()  # drift parameter
    if diff == None:
        diff = np.random.rand()*10  # diff coeff <pixels^2/frame>
    pos_out = [[] for x in range(n_particles*651+2)]
    pos_out[0] = ['Frame', 'Track_ID', 'X', 'Y', 'Type']
    m = 2
    for n in range(n_particles):
        if alpha == None:
            alpha = np.random.rand()*1.5+.25  # power : .25 to 1.75

        if alpha == 1.0:
            tag = 'anomalous : brown'
        elif alpha < 1.0:
            tag = 'anomalous : sub'
        else:
            tag = 'anomalous : super'

        n_time_steps = np.random.randint(5, time_max)  # Length of track

        x0 = size*np.random.uniform(size=(1, 2))

        wx = fbm_gauss(n_time_steps, alpha)
        wy = fbm_gauss(n_time_steps, alpha)

        x = np.zeros((n_time_steps, 1))
        y = np.zeros((n_time_steps, 1))
        x[0] = x0[0][0]
        y[0] = x0[0][1]

        for i in range(1, n_time_steps):
            x[i] = x[0] + i*mu + (diff**0.5)*wx[i]
            y[i] = y[0] + i*mu + (diff**0.5)*wy[i]
        for j in range(2, 653):
            if j-2 >= n_time_steps:
                pos_out[m] = [j-2, n+append_track, np.nan, np.nan, tag]
                m += 1
                continue
            pos_out[m] = [j-2, n+append_track, x[j-2][0], y[j-2][0], tag]
            m += 1
    output = StringIO()
    csv_writer = writer(output)
    for row in pos_out:
        csv_writer.writerow(row)
    output.seek(0)  # Get back to the begining of the StringIO
    df2 = pd.read_csv(output)
    return pd.concat([df, df2])

# Change this code!
def anomalous_diffusion_2(N=5001, alpha=1):
    n = np.linspace(-8, 48, 57)
    phix = 2*np.pi*np.random.rand(57)
    phiy = 2*np.pi*np.random.rand(57)
    t = np.linspace(0, N-1, N)
    xcomp = np.zeros(N)
    ycomp = np.zeros(N)
    for i, ti in enumerate(t):
        xcomp[i] = np.sum((np.cos(phix) - np.cos(np.sqrt(np.pi)**n*2*np.pi*ti/N + phix))/(np.sqrt(np.pi)**(n*alpha/2)))
        ycomp[i] = np.sum((np.cos(phiy) - np.cos(np.sqrt(np.pi)**n*2*np.pi*ti/N + phiy))/(np.sqrt(np.pi)**(n*alpha/2)))
    x = np.cumsum(np.diff(xcomp))
    y = np.cumsum(np.diff(ycomp))
    return x, y

def confined_diffusion(n_particles=1000, size=1600, l_trap=10, time_max=50, diff=None, vm=None, df=None):

   # Directed Diffusion
    # space_units == 'pixels'
    # time_units == 'frames'

    if type(df) == type(None) or (type(df) == type(pd.DataFrame()) and df.empty):
        df = pd.DataFrame(columns = ['Frame', 'Track_ID', 'X', 'Y', 'Type'])
        append_track = 0
    else:
        append_track = df['Track_ID'].max()+1

    if diff == None:
        diff = np.random.rand()*5  # diffusion coeff, <pixels^2/frame>
    if vm == None:
        vm = np.random.normal(loc=2, scale=1)  # mean velocity, pixel/frame

    dt = 1  # time step, frames

    k_temp = 4.2821e-21  # thermal energy, Boltzman x T @ 37ºC

    # Particle in a potential: settings the 'stiffness' of the energy potential
    ktrap = k_temp/l_trap**2  # = thermal energy / trap size ^ 2

    k = np.sqrt(2*diff*dt)

    pos_out = [[] for x in range(n_particles*651+2)] # Creating positional array
    pos_out[0] = ['Frame', 'Track_ID', 'X', 'Y', 'Type'] # First row is names
    m = 2

    for i in range(n_particles):
        n_time_steps = np.random.randint(5, time_max)
        # setting time variable
        time = np.array(range(0, n_time_steps))*dt
        # random initial position
        x0 = size*np.random.uniform(size=(1, 2))
        # set shape of displacement
        x = np.zeros(shape=(n_time_steps+1, 2))
        # initialize first step
        x[0] = x0
        for j in range(2, 653):
            if j-2 >= n_time_steps:
                pos_out[m] = [j-2, i+append_track, np.nan, np.nan, 'confined']
                m+=1
                continue

            dxtrap = diff/k_temp * -ktrap * (x[j-2] - x0) * dt  # displacement
            dxbrownian = k * np.random.randn(1, 2)
            x[j-1] = x[j-2] + dxtrap + dxbrownian
#             dxtrap = diff/k_temp * -ktrap * (x[j-1] - x0) * dt  # displacement
#             dxbrownian = k * np.random.randn(1, 2)
#             x[j] = x[j-1] + dxtrap + dxbrownian
            pos_out[m] = [j-2, i+append_track, x[j-2][0], x[j-2][1], 'confined']
            m+=1
    output = StringIO()
    csv_writer = writer(output)
    for row in pos_out:
        csv_writer.writerow(row)
    output.seek(0) # we need to get back to the start of the StringIO
    df2 = pd.read_csv(output)
    return pd.concat([df, df2])
