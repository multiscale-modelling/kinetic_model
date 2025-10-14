import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from sympy import symbols, Eq, solve
import numpy as np

mpl.rcParams['axes.linewidth'] = 4 #set the value globally
mpl.rcParams.update({'font.size': 24})

def find_line_equation(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def find_parabola_coeffs(point1, point2, y_max):
    x, a, h_x = symbols('x a h_x')
    
    x1, y1 = point1
    x2, y2 = point2
    
    y = a * (x - h_x)**2 + y_max
    
    eq1 = Eq(y.subs(x, x1), y1)
    eq2 = Eq(y.subs(x, x2), y2)
    
    solution = solve((eq1, eq2), (a, h_x))
    
    if len(solution) > 1:
        solution = [sol for sol in solution if sol[0] < 0][0]
    
    a_val, h_x_val = solution
    
    b_val = -2 * a_val * h_x_val
    c_val = a_val * h_x_val**2 + y_max
    
    return a_val, b_val, c_val
    
def get_energy_bar_data(start, val):
    x = np.linspace(start, start+1, 1000)
    y = x * 0 + val
    return np.array([x, y])

def get_inter_cnnct_data(start, end, val1, val2):
    m, b = find_line_equation(start, val1, end, val2)
    x = np.linspace(start, end, 1000)
    y = m * x + b
    return np.array([x, y])

def get_TS_curve_data(point1, point2, TS_val):
    try:
        a, b, c = find_parabola_coeffs(point1, point2, TS_val)
        x = np.linspace(point1[0], point2[0], 1000)
        y = a * x**2 + b * x + c
        return np.array([x, y])
    except TypeError:
        return get_inter_cnnct_data(point1[0], point2[0], point1[1], point2[1])
    
# Functions for repeated operations
def energy_bar(index, g_value):
    return get_energy_bar_data(index, g_value)

def inter_connect(index1, index2, g_value1, g_value2):
    return get_inter_cnnct_data(index1, index2, g_value1, g_value2)

def ts_curve(data1, data2, ts_value):
    return get_TS_curve_data(data1, data2, ts_value)
