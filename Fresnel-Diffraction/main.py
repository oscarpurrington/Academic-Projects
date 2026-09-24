# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 16:27:46 2025

@author: oscar
"""
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import time

# Variables #
LAMBDA = 6e-7  # Wave length [m]
K = 2 * np.pi / LAMBDA  # Wave Number [1/m]
E0 = 60  # Electric Field Strength [V/m]
Z =  0.05 # Observation distance [m]
aperture_size = 2e-5 # Aperture Size (width) [m]
xp1, xp2 = (  # Sets Boundaries of Aperture
    -aperture_size / 2,      #
    aperture_size / 2,       #
)                            #
yp1, yp2 = xp1, xp2
FACTOR = (K * E0) / (2 * np.pi * Z)  # Factor on Integral#
R = 1e-5  # Circular Aperture Radius [m]


def Fresnel2dreal(yp, xp, y, x, k, z):
    """Evaluates the Real part of the Integrand"""
    distance = (x - xp) ** 2 + (y - yp) ** 2
    phase = (k / (2 * z)) * distance

    return np.cos(phase)


def Fresnel2dimag(yp, xp, y, x, k, z):
    """Evaluates the Imaginary part of the Integrand"""
    distance = (x - xp) ** 2 + (y - yp) ** 2
    phase = (k / (2 * z)) * distance

    return np.sin(phase)


def ypfunc(xp):
    """Evaluates the lower y' limit of an x'."""
    return -np.sqrt(max(0, R**2 - xp**2))


def yp2func(xp):
    """Evaluates the upper y' limit of an x'."""
    return np.sqrt(max(0, R**2 - xp**2))


def part1():
    """Computes the diffraction pattern of light through a 2-d Aperture
    in 1 Dimension (as an intensity value), using a split-integration method
    using scipy.integrate.dblquad."""

    # Generates an array of numpoints x values
    numpoints = 100
    screen_size = 0.005
    # spaced over the screen size to contain the
    xvals = np.linspace(-screen_size, screen_size, numpoints)
    amplitudes = []                                     # area of interest.

    for x_pixel in xvals:                    # Loops each x value
        y_pixel = 0                             # Sets y to zero (1-d pattern)
        #
        real, real_error = integrate.dblquad(   # Parses integrate.dblquad to
            Fresnel2dreal,                      # obtain the real part of the
            # integral, and its error (dblquad
            xp1,
            xp2,                                # always returns this value)
            lambda xp: yp1,                     # The Lambda function returns
            lambda xp: yp2,                     # a constant upper and lower limit
            args=(y_pixel, x_pixel, K, Z),      # of integration, for any xp.
            #epsabs = 1e-10,
            #epsrel = 1e-10
        )                                       #
        #
        imag, imag_error = integrate.dblquad(   # Parses integrate.dblquad to
            Fresnel2dimag,                      # obtain the imaginary part
            xp1,                                # of the integral, in the same
            xp2,                                # manner as the real.
            lambda xp: yp1,                     #
            lambda xp: yp2,                     #
            args=(y_pixel, x_pixel, K, Z),      #
            #epsabs = 1e-10,
            #epsrel = 1e-10
        )                                       #

        # Calculates each intesnity value
        amplitude = (FACTOR * real) ** 2 + (FACTOR * imag) ** 2
        # Appends it to the array
        amplitudes.append(amplitude)

    plt.figure()                                      # Simple plotting function
    plt.plot(xvals, amplitudes, label="Intensity")  # to plot each intensity
    plt.title(                                        # against its corresponding
        "Fresnel Diffraction in One Dimension"        # x value
        f"\n from a Square Aperture (z={Z}m)")        #
    plt.xlabel("Screen Position")                     #
    plt.ylabel("Relative Intensity")                  #
    plt.legend()                                      #
    plt.plot()                                        #
    return


def part2():
    '''Computes the diffraction pattern of light through a rectangular 2-d aperture
    Note: Code was directly taken from exercise, with constants renamed for
    consistency. '''

    numpoints = 200
    Z = 0.1
    LAMBDA = 589e-9
    constant = np.pi*aperture_size/(LAMBDA*Z)
    x1 = -0.01
    x2 = -x1
    y1 = x1
    y2 = x2

    # Create 1-D arrays for screen coordinates
    xvals = np.linspace(x1, x2, numpoints)
    yvals = np.linspace(x1, x2, numpoints)

    # Generate two 2-D arrays representing the x and y values on the screen
    X, Y = np.meshgrid(xvals, yvals)

    # Compute ideal 2-D diffraction from aperture using the sinc function (Fraunhofer limit)
    # Note: 'constant' must be defined earlier (likely constant = np.pi * aperture_width / (wavelength * screen_distance))
    intensity = (np.sinc(constant * X) * np.sinc(constant * Y))**2

    # Define the limits for the plot
    extents = (x1, x2, y1, y2)

    # Plot the intensity array as an image
    plt.imshow(
        intensity,
        vmin=0.0,
        # Note: The original code has a typo here (vmax=1.0-intensity.max())
        vmax=1.0 * intensity.max(),
        extent=extents,
        origin="lower",
        cmap="nipy_spectral_r" # Example color map
    )

    # Add labels and title
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title('Rectangular diffraction;\nz = {:4.2f}m'.format(Z))
    plt.colorbar()
    plt.show()
    return


def part3():
    """Evaluates 2-d diffraction through a circular aperture by creating functions
    for the y values to form a circular aperture."""
    ###
    # Creates an array of numpoints x and y values within the screen size,
    # and initialises a 2-d array for the intensity values.
    ###
    time_init = time.time()
    numpoints = 50
    screen_size = 0.005
    xvals = np.linspace(-screen_size, screen_size, numpoints)
    yvals = xvals
    amplitudes = np.zeros((numpoints, numpoints))             

    ###
    # A nested for loop loops each value in the array, finding their intensity.
    # ypfunc() and yp2func() find the limits of integration for a given x, 
    # allowing a circle boundary to be found.
    ###
    for j, y in enumerate(yvals):
        for i, x in enumerate(xvals):
            real, real_error = integrate.dblquad(  
                Fresnel2dreal, -R, R, ypfunc, yp2func, args=(y, x, K, Z))
            imag, imag_error = integrate.dblquad(                         
                Fresnel2dimag, -R, R, ypfunc, yp2func, args=(y, x, K, Z))
            ###
            # Calculates the intensity and appends to the array
            ###
            amplitude = (FACTOR * real)**2 + (FACTOR * imag)**2
            amplitudes[j, i] = amplitude
        ###
        # Shows a progress bar, as runtimes can be quite long.
        ###
        print(f"progress: {j+1}/{numpoints}")

    # Plotting function taken from given code in part2().
    plt.figure()
    extents = (-screen_size, screen_size, -screen_size, screen_size)
    plt.imshow(
        amplitudes,
        vmin=0.0,
        vmax=1.0 * amplitudes.max(),
        extent=extents,
        origin="lower",
        cmap="nipy_spectral_r"
    )
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f'Circular Diffraction;\nz = {Z}m,\nnumpoints = {numpoints}')
    plt.colorbar()
    plt.show()
    print(f"runtime: {time.time()-time_init:4.2f}s")
    return


def part4():
    """"Approximates a 2-d circular diffraction pattern using a Monte Carlo
    approximation for the integral. This is much quicker than a direct integration
    but retains a lot of noise for smaller values of N and numpoints"""
    time_init = time.time()
    N = 10000             # Number of random samples per points.
    Area = (2*R)**2 # Area of square encompassing circluar region of interest
    numpoints = 200     # Number of pixels in the width of the square
    screen_size = 0.005   # Size of the screen

    xvals = np.linspace(-screen_size, screen_size,
                        numpoints)  # Generates arrays for
    yvals = xvals                   # x and y values, and
                                    # for intesnities.
    amplitudes = np.zeros((numpoints, numpoints))


    ###
    # Nested for loop loops each pixel, generating N random values for each.
    # Pixels with an r2 value <= to R**2 are counted, as they lie within our 
    # disc. The real and imaginary parts are computed, and then the in_aperture
    # mask applied. Amplitude is calculated as before and plotting function
    # from part2() used.
    ###
    for j, y in enumerate(yvals):                                
        for i, x in enumerate(xvals):                            
            x_random = np.random.uniform(-R, R, N)               
            y_random = np.random.uniform(-R, R, N)               
            r2 = x_random**2+y_random**2                         
            in_aperture = r2 <= R**2                             
            
            real = Fresnel2dreal(y_random, x_random, y, x, K, Z)
            imag = Fresnel2dimag(y_random, x_random, y, x, K, Z)
            
            real = real * in_aperture                            
            imag = imag * in_aperture                            
                                                         
            real_integral = Area * np.mean(real)                 
            imag_integral = Area * np.mean(imag)                 
                                                                 
            amplitude = ((FACTOR * real_integral)**2 +          
                (FACTOR * imag_integral)**2  )                     
            amplitudes[j, i] = amplitude                       
        print(f"progress: {j+1}/{numpoints}")                    

    plt.figure()
    extents = (-screen_size, screen_size, -screen_size, screen_size)
    plt.imshow(
        amplitudes,
        vmin=0.0,
        vmax=1.0 * amplitudes.max(),
        extent=extents,
        origin="lower",
        cmap="nipy_spectral_r"
    )
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f'Circular Diffraction (Monte Carlo);\nN={N},z={Z}')
    plt.colorbar()
    plt.show()
    print(f"runtime: {time.time()-time_init:4.2f}s")
    return


def main():
    """A simple text based menu to select programs.
    Program runs given choice once and then ends.
    """
    while True:
        print('''#---# Menu #---#
1: 1-d Diffraction from a 2-d Aperture
2: 2-d Diffraction from a Rectangular 2-d Aperture
3: 2-d Diffraction from a Circular 2-d Aperture
4: 2-d Diffraction from a Circular 2-d Aperture (Monte Carlo Approx.)
Q: Quit
              ''')
        choice = input('?: ').strip().lower()

        if choice == 'q':
            print('Closing...')
            break
        elif choice == '1':
            part1()
            break
        elif choice == '2':
            part2()
            break
        elif choice == '3':
            part3()
            break
        elif choice == '4':
            part4()
            break
        else:
            print("Invalid.")


main()
