# Modeling Comet Atmospheric Entry

This repo contains a small numerical study of a spherical comet entering Earth’s atmosphere. We model altitude-dependent air density, drag, gravity, and an optional “shaving”/ablation term that slowly reduces the comet’s radius. The goal is to see how **entry angle** and **initial size** affect drag over time and to estimate the **minimum (critical) starting radius** that survives to the ground.

**What’s here**
- A Jupyter notebook with the simulation and plots.
- (Optional) a `.py` export of the notebook for easy copy/paste.
- Figures used in the report (PNG).

**How it works**
We integrate the equations of motion with RK4, using $\rho(h)=\rho_0 e^{-h/H}$ and $F_D=\tfrac12 C_D \rho A v^2$; with ablation on, the radius decreases at a small constant rate.

**What to look at**
- Drag vs time for different entry angles and sizes.
- With vs without shaving (ablation) comparisons.
- Runs at the “critical” starting radius for each angle.

**Run it**
Open the notebook in Jupyter/VS Code and run all cells to reproduce the plots.

**Team**
Deividas Bilevicius · Jay Patel · Wiktor Kuzmiuk · Kyle Camden
