# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:30:30 2025

@author: OscarPurrington
"""
import time
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt

start_time = time.time()


"""CONSTANTS"""
G = 6.6743e-11  # Gravitational Constant (m^3.kg^-1.s^-2)
M_e = 5.972e24  # Earth mass (kg)
M_m = 7.348e22  # Moon mass (kg)
R_em = 384400e3  # Average Earth-Moon Seperation (m)
R_mp = 1.737e6 + 100e3  # Probe-Moon seperation [Moon radius + LLO(100km)] (m)
V_m = np.sqrt(G * M_e / R_em)  # Moon Orbital Speed (m.s^-1)
V_p = np.sqrt(G * M_m / R_mp)  # Probe Orbital Speed (m.s^-1)
T_m = 2360591.597  # Moon orbital period [1 sidereal month] (s)
t_span = (0, T_m)  # Time span for simulation [Multiple of sidereal months] (s)
t_coarse = np.linspace(
    t_span[0], t_span[1], 1001
)  # Simulation points for a coarser integration [suitable for moon-only orbit] (n/a)
t_fine = np.linspace(
    t_span[0], t_span[1], 5001
)  # Simulation points for moon-probe (n/a)


"""METHOD"""
def moon_orbit_function(t, state):
    x_m, y_m, v_mx, v_my = state

    a_mx = (-M_e * G * x_m) / ((x_m**2 + y_m**2) ** 1.5)
    a_my = (-M_e * G * y_m) / ((x_m**2 + y_m**2) ** 1.5)

    return (v_mx, v_my, a_mx, a_my)


def probe_orbit_function(t, state):

    # Considers probe in Moon Frame
    x_pm, y_pm, v_px, v_py = state  #

    a_px = (-M_m * G * x_pm) / (x_pm**2 + y_pm**2) ** 1.5
    a_py = (-M_m * G * y_pm) / (x_pm**2 + y_pm**2) ** 1.5

    return (v_px, v_py, a_px, a_py)


def plot(moon_orbit_solution, probe_orbit_solution, inp):
    axes = plt.axes()
    axes.set_aspect(1)
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title("Lunar orbit around Earth")

    plt.plot(0, 0, "o", color="green", markersize=8, label="Earth")

    if inp == "1":
        axes.plot(moon_orbit_solution.y[0], moon_orbit_solution.y[1], label="Moon")

        axes.legend(loc=1)
        plt.show()

    else:
        probe_x = moon_orbit_solution.y[0] + probe_orbit_solution.y[0]
        probe_y = moon_orbit_solution.y[1] + probe_orbit_solution.y[1]

        axes.plot(probe_x, probe_y, label="Probe")
        axes.plot(moon_orbit_solution.y[0], moon_orbit_solution.y[1], label="Moon")

        axes.legend(loc=1)
        plt.show()


def plot_zoom(moon_orbit_solution, probe_orbit_solution):
    axes = plt.axes()
    axes.set_aspect(1)
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title("View of Probe orbit around the Moon")

    # Full orbits
    axes.plot(
        moon_orbit_solution.y[0], moon_orbit_solution.y[1], label="Moon", color="orange"
    )
    axes.plot(
        moon_orbit_solution.y[0] + probe_orbit_solution.y[0],
        moon_orbit_solution.y[1] + probe_orbit_solution.y[1],
        label="Probe",
        color="blue",
    )
    axes.plot(0, 0, "o", color="green", markersize=8, label="Earth")

    axes.set_xlim(0, 1e8)
    axes.set_ylim(3.6e8, 4e8)

    axes.legend()
    plt.show()


def main():
    while True:
        print(
            """
            \n+++++ Menu +++++
            1: Calculate Moon Orbit
            2: Calculate Moon-Probe Orbital System
            Q: Quit\n
            """
        )
        inp = input("Enter Choice:  ").lower().strip()

        if inp == "q":
            print("\nClosing.\n")
            break
        elif inp == "1":
            start_time = time.time()

            state_init = (R_em, 0, 0, V_m)  # x_m, y_m, v_mx, v_my
            moon_orbit_solution = si.solve_ivp(
                moon_orbit_function,
                t_span,
                state_init,
                t_eval=t_coarse,
                rtol=1e-6,
                atol=1e-6,
            )
            plot(moon_orbit_solution, "", inp)

            print(f"\nTime Elapsed {time.time()-start_time:.3f}s\n")
        elif inp == "2":
            start_time = time.time()

            state_init_moon, state_init_probe = (R_em, 0, 0, V_m), (
                R_mp,
                0,
                0,
                V_p,
            )  # x_m, y_m, v_mx, v_my , x_p, y_p, v_px, v_py
            moon_orbit_solution = si.solve_ivp(
                moon_orbit_function,
                t_span,
                state_init_moon,
                t_eval=t_fine,
                rtol=1e-6,
                atol=1e-6,
            )
            probe_orbit_solution = si.solve_ivp(
                probe_orbit_function,
                t_span,
                state_init_probe,
                t_eval=t_fine,
                rtol=1e-6,
                atol=1e-6,
            )

            plot(moon_orbit_solution, probe_orbit_solution, inp)
            plot_zoom(moon_orbit_solution, probe_orbit_solution)

            print(f"\nTime Elapsed {time.time()-start_time:.3f}s\n")
        else:
            print("\nInvalid Choice\n")


main()
