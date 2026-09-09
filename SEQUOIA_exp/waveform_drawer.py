import matplotlib.pyplot as plt
import numpy as np
import gwsurrogate as gws
import time

from astropy.cosmology import Planck18, z_at_value
import astropy.units as u

from scripts.surrogate.sur_utils import DANSur

# ============================================================
# PARAMETERS. WRITE CUSTOM VALUES HERE
# ============================================================


q_dansur = 0.6 # Use DANSur convention: 0 < q_dansur < 1

#Spin vectors
chiA = np.array([
    0.0,
    0.0,
    0.0
])

chiB = np.array([
    0.0,
    0.0,
    0.0
])

chirp_mass_detector = 54.0     # Msun, chirp mass detected
dist_mpc = 4457.0               # Mpc, luminosity distance

f_low = 00.0                    # Hz,   minium frequency to be used in the generation of the gw
f_ref = 00.0                    # Hz    referece frequency to be used in the generation of the gw

dt = 1.0 / 4096.0               # s    temporal interval between waveform points

inclination = 0.0               #rad, inclination of the binary
phi_ref = 0.0                   #rad, polarization angle



# ============================================================
# Calculate total mass
# ============================================================

def calculate_total_mass(chirp_mass_detector, q_dansur, distance):
    """
    DANSur convention:
        q = m2 / m1

    NRH/NRSur convention:
        q = m1 / m2
    """

    q_nrh = 1.0 / q_dansur

    # Redshift
    z = z_at_value(
        Planck18.luminosity_distance,
        distance * u.Mpc
    )

    z = float(z)

    # Detector-frame -> source-frame chirp mass
    chirp_mass_source = chirp_mass_detector / (1.0 + z)

    # Total source-frame mass
    total_mass = (
        chirp_mass_source
        * (1.0 + q_nrh)**(6.0 / 5.0)
        / q_nrh**(3.0 / 5.0)
    )

    return total_mass, q_nrh, z




# ============================================================
# MASS AND MASS RATIO
# ============================================================

M, q_nrh, z = calculate_total_mass(
    chirp_mass_detector,
    q_dansur,
    dist_mpc
)

print("==============================================")
print("PHYSICAL PARAMETERS")
print("==============================================")
print("q DANSur (0 < q_dansur < 1) =", q_dansur)
print("q NRH/NRSur (q > 1) =", q_nrh)
print("Detector-frame chirp mass =", chirp_mass_detector, "Msun")
print("Redshift =", z)
print("Source-frame total mass =", M, "Msun")
print("Distance =", dist_mpc, "Mpc")
print("==============================================")


# ============================================================
# 1. NRH / NRHSur
# ============================================================

print("\nCalculating NRH/NRHSur...")
nrh = gws.LoadSurrogate("NRHybSur3dq8")

t0 = time.perf_counter()

domain_nrh, h_nrh, _ = nrh(
    q=q_nrh,
    chiA0=[0.0, 0.0, 0.0],
    chiB0=[0.0, 0.0, 0.0],
    M=M,
    dist_mpc=dist_mpc,
    f_low=f_low,
    inclination=inclination,
    f_ref=f_ref,
    phi_ref=phi_ref,
    mode_list=[(2, 2)],
    units="mks",
)

nrh_time = time.perf_counter() - t0

domain_nrh = np.asarray(domain_nrh)
h_nrh = np.asarray(h_nrh).squeeze()

print(f"NRH generation time: {nrh_time:.6f} s")
print("NRH samples:", len(domain_nrh))

# Extract h+
hplus_nrh = h_nrh.real


# ============================================================
# Centre NRH merger at t = 0
# ============================================================

domain_nrh = domain_nrh - domain_nrh[-1]


# ============================================================
# 2. NRSur7dq4
# ============================================================

print("\nCalculating NRSur7dq4...")

# ------------------------------------------------------------
# Convert physical parameters to dimensionless quantities
# ------------------------------------------------------------

M_sun_seconds = 4.92549095e-6

M_seconds = M * M_sun_seconds

f_low_dimensionless = M_seconds * f_low
f_ref_dimensionless = M_seconds * f_ref

dt_dimensionless = dt / M_seconds


# ------------------------------------------------------------
# Load NRSur7dq4
# ------------------------------------------------------------

t1 = time.perf_counter()

nr_surrogate = gws.LoadSurrogate("NRSur7dq4")

domain_nrsur, h_nrsur, _ = nr_surrogate(
    q=q_nrh,
    chiA0=chiA,
    chiB0=chiB,
    f_low=f_low_dimensionless,
    f_ref=f_ref_dimensionless,
    dt=dt_dimensionless,
    units="dimensionless",
)

nrsur_time = time.perf_counter() - t1

print(
    f"NRSur7dq4 generation time: {nrsur_time:.6f} s"
)

h22_nrsur = h_nrsur[(2, 2)]

# ------------------------------------------------------------
# Convert NRSur waveform to physical strain
#
# h_physical = (M / D) * h_dimensionless
# ------------------------------------------------------------

# Geometrical conversion:
# 1 Msun -> meters
M_sun_meters = 1476.6250385

# M in meters
M_meters = M * M_sun_meters

# 1 Mpc in meters
Mpc_meters = 3.0856775814913673e22

# Distance in meters
distance_meters = dist_mpc * Mpc_meters

# Dimensionless amplitude -> physical strain
conversion_factor = M_meters / distance_meters

hplus_nrsur = np.real(h22_nrsur) * conversion_factor

# Dimensionless time -> seconds
time_nrsur = np.asarray(domain_nrsur) * M_seconds

# Centre merger at t = 0
time_nrsur = time_nrsur - time_nrsur[-1]

print("NRSur samples:", len(time_nrsur))
print("NRSur amplitude conversion factor:", conversion_factor)


# ============================================================
# 3. DANSur
# ============================================================

print("\nCalculating DANSur...")

dansur = DANSur(device="cpu")

t2 = time.perf_counter()

domain, h, _  = dansur(
    q=q_dansur,
    chiA0=chiA,
    chiB0=chiB,
    M = M,
    dist_mpc=dist_mpc,
    f_low=f_low,
    inclination=inclination,
    f_ref=f_ref,
    phi_ref=phi_ref,
    units="mks",
)

dansur_time = time.perf_counter() - t2

domain = np.asarray(domain)
h = np.squeeze(np.asarray(h))

h_plus = np.real(h)
h_cross = np.imag(h)




# ============================================================
# Centre DANSur merger at t = 0
# ============================================================



# ============================================================
# PLOT 1 - NRH
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    domain_nrh,
    hplus_nrh
)

plt.xlabel(
    "Time [s]",
    fontsize=14
)

plt.ylabel(
    r"$h_+$",
    fontsize=14
)

plt.title(
    "NRH/NRHSur waveform",
    fontsize=16
)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "waveform_NRH.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT 2 - NRSur7dq4
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time_nrsur,
    hplus_nrsur
)

plt.xlabel(
    "Time [s]",
    fontsize=14
)

plt.ylabel(
    r"$h_+$",
    fontsize=14
)

plt.title(
    "NRSur7dq4 waveform",
    fontsize=16
)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "waveform_NRSur7dq4.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT 3 - DANSur
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    domain,
    h_plus
)

plt.xlabel(
    "Time [s]",
    fontsize=14
)

plt.ylabel(
    r"$h_+$",
    fontsize=14
)

plt.title(
    "DANSur waveform",
    fontsize=16
)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "waveform_DANSur.png",
    dpi=300
)

plt.show()


# ============================================================
# SUMMARY
# ============================================================

print()
print("==============================================")
print("WAVEFORMS GENERATED")
print("==============================================")

print("NRH:")
print("  Samples:", len(hplus_nrh))
print("  Time:", nrh_time, "s")
print("  Units: physical")

print()
print("NRSur7dq4:")
print("  Samples:", len(hplus_nrsur))
print("  Time:", nrsur_time, "s")
print("  Units: physical")

print()
print("DANSur:")
print("  Samples:", len(h_plus))
print("  Time:", dansur_time, "s")
print("  Units: physical")

print()
print("Saved files:")
print("  waveform_NRH.png")
print("  waveform_NRSur7dq4.png")
print("  waveform_DANSur.png")

print("==============================================")