from pandas import DataFrame
from pathlib import Path
from numpy import linspace
import logging

logging.basicConfig(format='%(levelname)-8s : %(message)s',
                    level=logging.INFO)
save_dir = "Csv"
file_name = "SCF_analitycal solution2"
Path(save_dir).mkdir(parents=True, exist_ok=True)

scale = 1.0e-3 # scale factor for the units, 1e-3 for [m], 1 for [mm]
G  = ((1/scale)**2)*50e3    # Shear modulus [MPa]

ν = 0.3  # Poisson's ratio
l  = scale * 30.0  # Characteristic length (bending) [mm]

N_values = linspace(0.1, 0.9, 9)  # list of values for coupling numer N
D = l  # hole diamater
R = 0.5 * D  # radius of the hole
# Analytical solution
def AnalyticalSolution(ν , l, c, R):
  # Modified Bessel function of the second kind of real order v :
  from scipy.special import kv
  F = 8.0*(1.0-ν )*((l**2)/(c**2)) * \
  		1.0 / (( 4.0 + ((R**2)/(c**2)) + \
  		((2.0*R)/c) * kv(0,R/c)/kv(1,R/c) ))
  SCF = (3.0 + F) / (1.0 + F) # stress concentration factor
  return (SCF)

SCF_list = []
for N in N_values:
    c = l / N
    SCF = AnalyticalSolution(ν, l, c, R)
    logging.info(f"Stress Concentration Factor (SCF) for N={N:.2f}: {SCF:.4f}")
    SCF_list.append(SCF)

df = DataFrame({'N': N_values,
                'SCF': SCF_list})
df.to_csv(path:="".join([save_dir, "/", file_name]), index=False)
logging.info(f"Saving results to {path}")
