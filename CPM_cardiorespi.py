import numpy as np
import matplotlib.pyplot as plt

# ========================================================================================
# SIMULATION PARAMETERS
# ========================================================================================
tmin, tmax, T = 0.0, 15.0, 0.0002 # Reduced T for respiratory model stability
t_eval = np.arange(tmin, tmax, T)
N = len(t_eval)

# ========================================================================================
# CARDIOVASCULAR MODEL PARAMETERS
# ========================================================================================
HR = 70
TBV = 4750

# Cardiovascular Elastances, Resistances, and Volumes
elastance_orig = np.array([
    [1.43, np.nan], [0.6, np.nan], [0.0169, np.nan], [0.0182, np.nan],
    [0.05, 0.15], [0.057, 0.49], [0.233, np.nan], [0.0455, np.nan],
    [0.12, 0.28], [0.09, 4]
])
elastance = elastance_orig.T
resistance = np.array([0.06, 0.85, 0.09, 0.003, 0.003, 0.003, 0.11, 0.003, 0.003, 0.008])
uvolume = np.array([140, 370, 1000, 1190, 14, 26, 50, 350, 11, 20])

# Initialize cardiovascular state variables
V = TBV * (uvolume / uvolume.sum())
P = np.zeros(10) # Current cardiac pressures
F = np.zeros(10) # Current cardiac flows
ncc = 1 # Counter for cardiac cycle steps

# ========================================================================================
# RESPIRATORY MODEL PARAMETERS
# ========================================================================================
RR = 12 # Respiratory rate (breaths per minute)
RRP = 60 / RR # Respiratory period (seconds)

# Respiratory circuit parameters
C_l, C_tr, C_b, C_A = 0.00127, 0.00238, 0.0131, 0.2  # Capacitances (L/cmH2O)
R_ml, R_lt, R_tb, R_bA = 1.021 , 0.3369, 0.3063, 0.0817 # Resistances (cmH2O*s/L)

# Unstressed volumes
UV_l = 34.4/1000  # L (converted from ml)
UV_tr = 6.63/1000  # L
UV_b = 18.7/1000  # L
UV_A = 1.263  # L

# Respiratory muscle parameters
Pmus_min, IEratio = -5.0, 1.0 # Min muscle pressure (cmH2O), I:E ratio
T_breath = 60 / RR # Breath period (s)
TI = T_breath * IEratio / (1 + IEratio) # Inspiratory time (s)
TE = T_breath - TI # Expiratory time (s)

# Initialize respiratory state variables
P_resp = np.zeros((4, N)) # Respiratory pressures [Pl, Ptr, Pb, PA] (cmH2O)
V_resp = np.zeros((4, N)) # Respiratory volumes [Vl, Vtr, Vb, VA] (L)
F_resp = np.zeros((4, N)) # Respiratory flows [Vdot_ml, Vdot_lt, Vdot_tb, Vdot_bA] (L/s)
Pmus_current = 0.0 # Initialize Pmus for integration

# Initialize respiratory volumes at unstressed volumes (FRC - Functional Residual Capacity)
V_resp[0, 0] = UV_l      # Initial larynx volume
V_resp[1, 0] = UV_tr     # Initial trachea volume  
V_resp[2, 0] = UV_b      # Initial bronchi volume
V_resp[3, 0] = UV_A

# ========================================================================================
# OUTPUT ARRAYS INITIALIZATION
# ========================================================================================
output1 = np.zeros(N) # ela
output2 = np.zeros(N) # elv
output3 = np.zeros(N) # era
output4 = np.zeros(N) # erv
outputPTH = np.zeros(N) # Pintra (derived from P_alv)
outputHR = np.zeros(N) # HR (can be dynamic if HR changes)
outputV = np.zeros((10, N))
outputP = np.zeros((10, N))
outputF = np.zeros((10, N))
outputPmus = np.zeros(N) # To store Pmus values

# ========================================================================================
# MAIN SIMULATION LOOP
# ========================================================================================
for n in range(N):
    t_current = n * T

    # --------------------------------------------------------------------------------
    # RESPIRATORY MODEL UPDATE
    # --------------------------------------------------------------------------------
    
    # 1) Respiratory muscle pressure derivative (dPmus_dt)
    T_resp = 60 / RR                 # breath period (s)
    TI = T_resp * IEratio / (1 + IEratio)
    TE = T_resp - TI
    exp_time = TE / 5
    cycle_time = t_current % T_resp

    if 0 <= cycle_time <= TI:
        dPmus_dt = 2 * (-Pmus_min / (TI * TE)) * cycle_time + (Pmus_min * T_resp) / (TI * TE)
    else:
        dPmus_dt = (
            -Pmus_min
            / (exp_time * (1 - np.exp(-TE / exp_time)))
        ) * np.exp(-(cycle_time - TI) / exp_time)

    # 2) Integrate dPmus_dt → Pmus_current (Euler)
    if n > 0:
        Pmus_current = Pmus_current + T * dPmus_dt

    outputPmus[n] = Pmus_current  # store muscle pressure

    # 3) Calculate respiratory flows using current pressures
    Vdot_ml = (0 - P_resp[0, n]) / R_ml  # Flow from atmosphere to larynx
    Vdot_lt = (P_resp[0, n] - P_resp[1, n]) / R_lt    # Flow from larynx to trachea
    Vdot_tb = (P_resp[1, n] - P_resp[2, n]) / R_tb    # Flow from trachea to bronchi
    Vdot_bA = (P_resp[2, n] - P_resp[3, n]) / R_bA    # Flow from bronchi to alveoli

    F_resp[0, n] = Vdot_ml
    F_resp[1, n] = Vdot_lt
    F_resp[2, n] = Vdot_tb
    F_resp[3, n] = Vdot_bA

    # 4) Update volumes via dV/dt = Flow_in - Flow_out for next time step
    if n < N-1:  # Only update if not the last time step
        dV0_dt = Vdot_ml - Vdot_lt
        dV1_dt = Vdot_lt - Vdot_tb
        dV2_dt = Vdot_tb - Vdot_bA
        dV3_dt = Vdot_bA  # alveoli is terminal

        V_resp[0, n+1] = V_resp[0, n] + T * dV0_dt
        V_resp[1, n+1] = V_resp[1, n] + T * dV1_dt
        V_resp[2, n+1] = V_resp[2, n] + T * dV2_dt
        V_resp[3, n+1] = V_resp[3, n] + T * dV3_dt

        # 5) Calculate pressures from volumes using P = (V - UV) / C
        P_resp[0, n+1] = (V_resp[0, n+1] - UV_l) / C_l
        P_resp[1, n+1] = (V_resp[1, n+1] - UV_tr) / C_tr
        P_resp[2, n+1] = (V_resp[2, n+1] - UV_b) / C_b
        P_resp[3, n+1] = ((V_resp[3, n+1] - UV_A) / C_A) + Pmus_current

    # 6) Convert alveolar pressure to mmHg for cardiovascular coupling
    P_alv_cmH2O = P_resp[3,n]
    Pintra_mmHg = P_alv_cmH2O * 0.73556 # Convert cmH2O to mmHg

    # --------------------------------------------------------------------------------
    # CARDIOVASCULAR MODEL UPDATE
    # --------------------------------------------------------------------------------
    
    # 1) Initialize cardiac cycle parameters at start of each cycle
    if ncc==1:
        HP=60/HR;
        Tas=0.03+0.09*HP;
        Tav=0.01;
        Tvs=0.16+0.2*HP;
    
    # 2) Calculate atrial elastances
    if ncc <= round(Tas/T):
        aaf = np.sin(np.pi*(ncc-1)*T/Tas);
    else:
        aaf=0;

    ela=elastance[0,8]+(elastance[1,8]-elastance[0,8])*aaf; # Left Atrium (comp 8)
    era=elastance[0,4]+(elastance[1,4]-elastance[0,4])*aaf; # Right Atrium (comp 4)

    # 3) Calculate ventricular elastances
    if ncc <= round((Tas+Tav)/T):
        vaf=0;
    elif ncc <= round((Tas+Tav+Tvs)/T):
        vaf=np.sin(np.pi*((ncc-1)*T-(Tas+Tav))/Tvs);
    else:
        vaf=0;

    elv=elastance[0,9]+(elastance[1,9]-elastance[0,9])*vaf; # Left Ventricle (comp 9)
    erv=elastance[0,5]+(elastance[1,5]-elastance[0,5])*vaf; # Right Ventricle (comp 5)

    # 4) Calculate cardiovascular pressures (using Pintra_mmHg from respiratory model)
    P[0]=elastance[0,0]*(V[0]-uvolume[0])+Pintra_mmHg;
    P[1]=elastance[0,1]*(V[1]-uvolume[1]);
    P[2]=elastance[0,2]*(V[2]-uvolume[2]);
    P[3]=elastance[0,3]*(V[3]-uvolume[3])+Pintra_mmHg;
    P[4]=era*(V[4]-uvolume[4])+Pintra_mmHg;
    P[5]=erv*(V[5]-uvolume[5])+Pintra_mmHg;
    P[6]=elastance[0,6]*(V[6]-uvolume[6])+Pintra_mmHg;
    P[7]=elastance[0,7]*(V[7]-uvolume[7])+Pintra_mmHg;
    P[8]=ela*(V[8]-uvolume[8])+Pintra_mmHg;
    P[9]=elv*(V[9]-uvolume[9])+Pintra_mmHg;

    # 5) Calculate flows with valve logic
    F[0]=(P[0]-P[1])/resistance[0]; # Flow from compartment 0 to 1
    F[1]=(P[1]-P[2])/resistance[1]; # Flow from compartment 1 to 2
    F[2]=(P[2]-P[3])/resistance[2]; # Flow from compartment 2 to 3

    if P[3]>P[4]: # Valve with backflow resistance
        F[3]=(P[3]-P[4])/resistance[3];
    else:
        F[3]=(P[3]-P[4])/(10*resistance[3]);

    if P[4]>P[5]: # Atrioventricular valve
        F[4]=(P[4]-P[5])/resistance[4];
    else:
        F[4]=0; # No backflow

    if P[5]>P[6]: # Ventriculo-arterial valve
        F[5]=(P[5]-P[6])/resistance[5];
    else:
        F[5]=0; # No backflow

    F[6]=(P[6]-P[7])/resistance[6]; # Flow from compartment 6 to 7

    if P[7]>P[8]: # Valve with backflow resistance
        F[7]=(P[7]-P[8])/resistance[7];
    else:
        F[7]=(P[7]-P[8])/(10*resistance[7]);

    if P[8]>P[9]: # Atrioventricular valve
        F[8]=(P[8]-P[9])/resistance[8];
    else:
        F[8]=0; # No backflow

    if P[9]>P[0]: # Ventriculo-arterial valve
        F[9]=(P[9]-P[0])/resistance[9];
    else:
        F[9]=0; # No backflow

    # 6) Update cardiovascular volumes based on flows
    V[0]=V[0]+T*(F[9]-F[0]);
    V[1]=V[1]+T*(F[0]-F[1]);
    V[2]=V[2]+T*(F[1]-F[2]);
    V[3]=V[3]+T*(F[2]-F[3]);
    V[4]=V[4]+T*(F[3]-F[4]);
    V[5]=V[5]+T*(F[4]-F[5]);
    V[6]=V[6]+T*(F[5]-F[6]);
    V[7]=V[7]+T*(F[6]-F[7]);
    V[8]=V[8]+T*(F[7]-F[8]);
    V[9]=V[9]+T*(F[8]-F[9]);

    # 7) Update cardiac cycle counter
    ncc=ncc+1;
    if ncc==1+round(HP/T):   # if end of cycle
        ncc=1;                # start new one

    # --------------------------------------------------------------------------------
    # STORE OUTPUT DATA
    # --------------------------------------------------------------------------------
    output1[n]=ela;
    output2[n]=elv;
    output3[n]=era;
    output4[n]=erv;
    outputPTH[n]=Pintra_mmHg;
    outputHR[n]=HR;
    outputV[:,n]=V;
    outputP[:,n]=P;
    outputF[:,n]=F;

# ========================================================================================
# PLOTTING AND VISUALIZATION
# ========================================================================================
# Remove first 5 seconds from plots
start_idx = int(5.0 / T)  # Index corresponding to 5 seconds
t_plot = t_eval[start_idx:]

# Create two separate figures
fig1, axs1 = plt.subplots(2, 2, figsize=(16, 10))

# Plot Left Heart PV Loop (Top-Left)
ax_lv_pv = axs1[0, 0]
ax_lv_pv.plot(outputV[9,start_idx:], outputP[9,start_idx:], 'r-', label='Left Ventricle', linewidth=2)
ax_lv_pv.plot(outputV[5,start_idx:], outputP[5,start_idx:], 'b-', label='Right ventricle', linewidth=2)
ax_lv_pv.set_xlabel('Volume (ml)')
ax_lv_pv.set_ylabel('Pressure (mmHg)')
ax_lv_pv.legend()
ax_lv_pv.set_title('Left and Right Heart PV Loops')
ax_lv_pv.grid(True)


# Plot Cardiovascular Pressures - Arterial, Pulmonary, Venous (Top-Right)
ax_cardio = axs1[0, 1]
ax_cardio.plot(t_plot, outputP[0,start_idx:], 'r-', label='Aorta (Arterial)', linewidth=2)
ax_cardio.plot(t_plot, outputP[6,start_idx:], 'g-', label='Pulmonary Artery', linewidth=2)
ax_cardio.plot(t_plot, outputP[7,start_idx:], 'c-', label='Vena Cava', linewidth=1)
ax_cardio.set_xlabel('Time (s)')
ax_cardio.set_ylabel('Pressure (mmHg)')
ax_cardio.legend()
ax_cardio.set_title('Arterial, Pulmonary & Venous Pressures')
ax_cardio.grid(True)

# Plot Combined Respiratory Variables (Bottom-Left)
ax_resp_combined = axs1[1, 0]
V_deadspace = (V_resp[0,start_idx:] + V_resp[1,start_idx:] + V_resp[2,start_idx:]) * 1000
V_alveolar = V_resp[3,start_idx:] * 1000
V_total_lung = V_deadspace + V_alveolar

# Create respiratory PV loop
ax_resp_combined.plot(V_total_lung, P_resp[3,start_idx:], label='Respiratory PV Loop', color='black', linewidth=2)
ax_resp_combined.set_xlabel('Total Lung Volume (ml)')
ax_resp_combined.set_ylabel('Alveolar Pressure (cmH2O)')
ax_resp_combined.legend()
ax_resp_combined.set_title('Respiratory PV Loop')
ax_resp_combined.grid(True)

# Plot Respiratory Flow (Bottom-Right)
ax_flows = axs1[1, 1]
ax_flows.plot(t_plot, F_resp[0,start_idx:]*1000, label='Mouth-Larynx Flow (ml/s)', 
             color='red', linewidth=2)
ax_flows.set_xlabel('Time (s)')
ax_flows.set_ylabel('Flow (ml/s)')
ax_flows.legend()
ax_flows.set_title('Respiratory Flow')
ax_flows.grid(True)

plt.tight_layout()
plt.show()

# Save main figure and individual subplots
output_dir = '/Users/l.m.vanloon/Library/CloudStorage/OneDrive-UniversityofTwente/Manuscripts/Model tutorial/images/'
fig1.savefig(f'{output_dir}cardiorespiratory_main_plots.png', dpi=300, bbox_inches='tight')

# Save individual subplots as separate files
# Cardiac PV Loops
fig_cardiac_pv, ax_cardiac_pv = plt.subplots(figsize=(8, 6))
ax_cardiac_pv.plot(outputV[9,start_idx:], outputP[9,start_idx:], 'r-', label='Left Ventricle', linewidth=2)
ax_cardiac_pv.plot(outputV[5,start_idx:], outputP[5,start_idx:], 'b-', label='Right ventricle', linewidth=2)
ax_cardiac_pv.set_xlabel('Volume (ml)')
ax_cardiac_pv.set_ylabel('Pressure (mmHg)')
ax_cardiac_pv.legend()
ax_cardiac_pv.set_title('Left and Right Heart PV Loops')
ax_cardiac_pv.grid(True)
fig_cardiac_pv.savefig(f'{output_dir}cardiac_pv_loops.png', dpi=300, bbox_inches='tight')
plt.close(fig_cardiac_pv)

# Cardiovascular Pressures
fig_cardio_press, ax_cardio_press = plt.subplots(figsize=(8, 6))
ax_cardio_press.plot(t_plot, outputP[0,start_idx:], 'r-', label='Aorta (Arterial)', linewidth=2)
ax_cardio_press.plot(t_plot, outputP[6,start_idx:], 'g-', label='Pulmonary Artery', linewidth=2)
ax_cardio_press.plot(t_plot, outputP[7,start_idx:], 'c-', label='Vena Cava', linewidth=1)
ax_cardio_press.set_xlabel('Time (s)')
ax_cardio_press.set_ylabel('Pressure (mmHg)')
ax_cardio_press.legend()
ax_cardio_press.set_title('Arterial, Pulmonary & Venous Pressures')
ax_cardio_press.grid(True)
fig_cardio_press.savefig(f'{output_dir}cardiovascular_pressures.png', dpi=300, bbox_inches='tight')
plt.close(fig_cardio_press)

# Respiratory PV Loop
fig_resp_pv, ax_resp_pv = plt.subplots(figsize=(8, 6))
V_deadspace = (V_resp[0,start_idx:] + V_resp[1,start_idx:] + V_resp[2,start_idx:]) * 1000
V_alveolar = V_resp[3,start_idx:] * 1000
V_total_lung = V_deadspace + V_alveolar
ax_resp_pv.plot(V_total_lung, P_resp[3,start_idx:], label='Respiratory PV Loop', color='black', linewidth=2)
ax_resp_pv.set_xlabel('Total Lung Volume (ml)')
ax_resp_pv.set_ylabel('Alveolar Pressure (cmH2O)')
ax_resp_pv.legend()
ax_resp_pv.set_title('Respiratory PV Loop')
ax_resp_pv.grid(True)
fig_resp_pv.savefig(f'{output_dir}respiratory_pv_loop.png', dpi=300, bbox_inches='tight')
plt.close(fig_resp_pv)

# Respiratory Flow
fig_resp_flow, ax_resp_flow = plt.subplots(figsize=(8, 6))
ax_resp_flow.plot(t_plot, F_resp[0,start_idx:]*1000, label='Mouth-Larynx Flow (ml/s)', 
                 color='red', linewidth=2)
ax_resp_flow.set_xlabel('Time (s)')
ax_resp_flow.set_ylabel('Flow (ml/s)')
ax_resp_flow.legend()
ax_resp_flow.set_title('Respiratory Flow')
ax_resp_flow.grid(True)
fig_resp_flow.savefig(f'{output_dir}respiratory_flow.png', dpi=300, bbox_inches='tight')
plt.close(fig_resp_flow)

# Second figure: Driver signals
fig2, axs2 = plt.subplots(1, 2, figsize=(16, 5))

# Plot Cardiac Elastances (Left)
ax_cardiac = axs2[0]
ax_cardiac.plot(t_plot, output1[start_idx:], 'r--', label='Left Atrial Elastance', linewidth=2)
ax_cardiac.plot(t_plot, output2[start_idx:], 'r-', label='Left Ventricular Elastance', linewidth=2)
ax_cardiac.plot(t_plot, output3[start_idx:], 'b--', label='Right Atrial Elastance', linewidth=1)
ax_cardiac.plot(t_plot, output4[start_idx:], 'b-', label='Right Ventricular Elastance', linewidth=1)
ax_cardiac.set_xlabel('Time (s)')
ax_cardiac.set_ylabel('Elastance (mmHg/ml)')
ax_cardiac.legend()
ax_cardiac.set_title('Cardiac Elastance Drivers')
ax_cardiac.grid(True)

# Plot Respiratory Muscle Pressure (Right)
ax_respiratory = axs2[1]
ax_respiratory.plot(t_plot, outputPmus[start_idx:], 'cyan', linewidth=2)
ax_respiratory.set_xlabel('Time (s)')
ax_respiratory.set_ylabel('Pressure (cmH2O)')
ax_respiratory.set_title('Respiratory Driver (Pmus)')
ax_respiratory.grid(True)

plt.tight_layout()
plt.show()

# Save driver signals figure and individual plots
fig2.savefig(f'{output_dir}driver_signals.png', dpi=300, bbox_inches='tight')

# Save individual driver plots
# Cardiac Elastances
fig_cardiac_elast, ax_cardiac_elast = plt.subplots(figsize=(8, 6))
ax_cardiac_elast.plot(t_plot, output1[start_idx:], 'r--', label='Left Atrial Elastance', linewidth=2)
ax_cardiac_elast.plot(t_plot, output2[start_idx:], 'r-', label='Left Ventricular Elastance', linewidth=2)
ax_cardiac_elast.plot(t_plot, output3[start_idx:], 'b--', label='Right Atrial Elastance', linewidth=1)
ax_cardiac_elast.plot(t_plot, output4[start_idx:], 'b-', label='Right Ventricular Elastance', linewidth=1)
ax_cardiac_elast.set_xlabel('Time (s)')
ax_cardiac_elast.set_ylabel('Elastance (mmHg/ml)')
ax_cardiac_elast.legend()
ax_cardiac_elast.set_title('Cardiac Elastance Drivers')
ax_cardiac_elast.grid(True)
fig_cardiac_elast.savefig(f'{output_dir}cardiac_elastances.png', dpi=300, bbox_inches='tight')
plt.close(fig_cardiac_elast)

# Respiratory Muscle Pressure
fig_resp_driver, ax_resp_driver = plt.subplots(figsize=(8, 6))
ax_resp_driver.plot(t_plot, outputPmus[start_idx:], 'cyan', linewidth=2)
ax_resp_driver.set_xlabel('Time (s)')
ax_resp_driver.set_ylabel('Pressure (cmH2O)')
ax_resp_driver.set_title('Respiratory Driver (Pmus)')
ax_resp_driver.grid(True)
fig_resp_driver.savefig(f'{output_dir}respiratory_muscle_pressure.png', dpi=300, bbox_inches='tight')
plt.close(fig_resp_driver)

print("All plots saved as individual PNG files:")
print("- cardiorespiratory_main_plots.png")
print("- cardiac_pv_loops.png")
print("- cardiovascular_pressures.png") 
print("- respiratory_pv_loop.png")
print("- respiratory_flow.png")
print("- driver_signals.png")
print("- cardiac_elastances.png")
print("- respiratory_muscle_pressure.png")
