import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

# --- Parameters ---
g = 9.81
D = 10.0
nx = 200
x_start, x_end = 0.0, 5.0
x = np.linspace(x_start, x_end, nx)
dx = x[1] - x[0]
t_start, t_stop = 0.0, 1.0
cf = 0.5  # friction coefficient

# --- Bed profile (wavy) ---
zb = -D + 0.4 * np.sin(2 * np.pi * x / x_end * ((nx - 1)/nx) * 5)

# --- Initial condition ---
def initial_conditions():
    h0 = 1 * np.exp(-100 * ((x / x_end - 0.5) * x_end)**2) - zb
    q0 = np.zeros_like(h0)
    return np.concatenate([h0, q0])

# --- GABC treatment at domain ends ---
def apply_gabc(h, q, dhdt, dqdt):
    # Left boundary x=0
    c = np.sqrt(g * max(h[0], 1e-6))
    dqdt_in = (1 / h[0]) * (dqdt[0] - (q[0]/h[0]) * dhdt[0] + c * dhdt[0])
    dqdt_out = (1 / h[0]) * (dqdt[0] - (q[0]/h[0]) * dhdt[0] - c * dhdt[0])
    dqdt[0] = dqdt_in + dqdt_out  # superpose characteristics (simplified)

    # Right boundary x=L
    c = np.sqrt(g * max(h[-1], 1e-6))
    dqdt_in = (1 / h[-1]) * (dqdt[-1] - (q[-1]/h[-1]) * dhdt[-1] - c * dhdt[-1])
    dqdt_out = (1 / h[-1]) * (dqdt[-1] - (q[-1]/h[-1]) * dhdt[-1] + c * dhdt[-1])
    dqdt[-1] = dqdt_in + dqdt_out  # superpose characteristics (simplified)

# --- Time derivative function ---
def shallow_water_rhs(t, u):
    h, q = u[:nx], u[nx:]
    ζ = h + zb
    h_p, h_m = np.roll(h, -1), np.roll(h, 1)
    q_p, q_m = np.roll(q, -1), np.roll(q, 1)
    ζ_p, ζ_m = np.roll(ζ, -1), np.roll(ζ, 1)

    dhdt = -(q_p - q_m) / (2 * dx)
    dqdx = ((q_p**2 / (h_p + 1e-6)) - (q_m**2 / (h_m + 1e-6))) / (2 * dx)
    dzetadx = (ζ_p - ζ_m) / (2 * dx)
    dqdt = -dqdx - g * h * dzetadx - cf * q * np.abs(q) / (h**2 + 1e-6)

    apply_gabc(h, q, dhdt, dqdt)
    return np.concatenate([dhdt, dqdt])

# --- Solve the PDE ---
u0 = initial_conditions()
t_eval = np.linspace(t_start, t_stop, 500)
# sol = solve_ivp(shallow_water_rhs, (t_start, t_stop), u0, method='RK45',
#                 t_eval=t_eval, rtol=1e-6, atol=1e-6)
sol = solve_ivp(shallow_water_rhs, (t_start, t_stop), u0,
                method='DOP853', t_eval=t_eval,
                rtol=1e-8, atol=1e-8)

# --- Create enhanced animation ---
fig, ax = plt.subplots(figsize=(12, 6))
line_h, = ax.plot([], [], label='Free surface', color='blue', lw=2)
line_bed, = ax.plot([], [], label='Bed elevation', color='black', lw=1)
fill = None
ax.set_xlim(x_start, x_end)
ax.set_ylim(np.min(zb)-0.1, np.max(sol.y[:nx]) + 0.2)
ax.set_xlabel("x [m]")
ax.set_ylabel("Elevation [m]")
ax.set_title("1D Shallow Water Simulation with GABC")
ax.grid(True)
ax.legend(loc='upper right')

def init():
    global fill
    line_h.set_data([], [])
    line_bed.set_data(x, zb)
    fill = ax.fill_between(x, zb, zb, color='lightblue', alpha=0.5)
    return line_h, line_bed, fill

def update(frame):
    global fill
    for coll in ax.collections:
        coll.remove()
    h = sol.y[:nx, frame]
    surface = h + zb
    line_h.set_data(x, surface)
    fill = ax.fill_between(x, zb, surface, color='lightblue', alpha=0.5)
    ax.set_title(f"t = {sol.t[frame]:.2f} s")
    return line_h, line_bed, fill

ani = FuncAnimation(fig, update, frames=len(sol.t), init_func=init, blit=False)
ani.save("shallow_water_gabc_nice_rk8.gif", writer=PillowWriter(fps=30))
plt.close()

print("✅ Enhanced GIF saved as 'shallow_water_gabc_nice.gif'")