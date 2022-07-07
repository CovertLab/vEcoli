import os
import scipy.integrate
from matplotlib import pyplot as plt

from vivarium.library.units import units

def ficks_law(t, y, p):
    c_p, c_c = y
    v_p, p_o, a_o, c_o, p_i, a_i, v_c, v_max, k_m = p
    
    # Note that c_p and c_c require units to be tacked on here
    f = [(1/v_p * (p_o * a_o * (c_o - c_p*units.uM/2) - p_i * a_i * (c_p*units.uM - c_c*units.uM/2))).to(units.uM / units.sec).magnitude,
         (1/v_c * (p_i * a_i * (c_p*units.uM - c_c*units.uM/2) - (v_max * c_c*units.uM/(k_m + c_c*units.uM)))).to(units.uM / units.sec).magnitude]
    return f

# Diffusion parameters
v_p = 0.24 * units.fL
p_o = 0.7E-7 * units.cm / units.sec
a_o = 4.52 * units.um**2
# 1.5 mg/L tetracycline
c_o = 20 * units.uM
p_i = 3E-6 * units.cm / units.sec
a_i = 4.20 * units.um**2
v_c = 0.96 * units.fL
v_max = 0.2 * units.nmol / units.mg / units.min * (351.1 * units.fg)
k_m = 200 * units.uM
p = [v_p, p_o, a_o, c_o, p_i, a_i, v_c, v_max, k_m]

# Periplasmic and cytoplasmic concentrations
c_p = 0
c_c = 0
y0 = [c_p, c_c]

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 1200.0
numpoints = 250

# Create the time samples for the output of the ODE solver.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

sol = scipy.integrate.solve_ivp(ficks_law, [0, 1200], y0, t_eval=t, args=(p,), atol=abserr, rtol=relerr)
print(sol.y)
plt.plot(t, sol.y[0])
plt.plot(t, sol.y[1])
if os.path.exists('out/'):
    plt.savefig('out/odesol_wet.png')
