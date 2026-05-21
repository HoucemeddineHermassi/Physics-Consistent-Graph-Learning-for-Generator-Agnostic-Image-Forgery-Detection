import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# --- Academic Configuration ----------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Palatino']
plt.rcParams['mathtext.fontset'] = 'cm'

# Multi-row layout (Optimized for zero overlap)
fig, ax = plt.subplots(figsize=(14, 12), facecolor='white')
ax.set_axis_off()
ax.set_xlim(0, 12)
ax.set_ylim(-3.5, 15.5) # Increased upper bound

# --- Premium Palette (Deep Academic Contrast) ----------------
C_NAVY_DEEP  = '#0F172A'
C_NAVY_MEDIUM= '#334155'
C_CRIMSON    = '#991B1B'
C_PEACH      = '#FEF2F2'
C_EMERALD    = '#065F46'
C_MINT       = '#ECFDF5'
C_SLATE      = '#475569'
C_AMBER      = '#92400E'
C_GOLD       = '#D97706'
C_WHITE      = '#FFFFFF'

# --- Drawing Helpers with Strict Spacing ---------------------
def draw_node(ax, x, y, label, sublabel='', color=C_NAVY_DEEP, bg=C_WHITE, radius=0.5, shape='circle', sub_pos='below'):
    # Subtle drop shadow
    shadow = mpatches.Circle((x+0.04, y-0.04), radius, facecolor='#E2E8F0', zorder=9, alpha=0.5)
    ax.add_patch(shadow)

    if shape == 'circle':
        patch = mpatches.Circle((x, y), radius, linewidth=2.5, edgecolor=color, facecolor=bg, zorder=10)
    else: # Diamond for exogenous/noise
        patch = mpatches.RegularPolygon(xy=(x, y), numVertices=4, radius=radius*1.1, 
                                        orientation=0, linewidth=1.5, edgecolor=color, 
                                        facecolor=bg, zorder=10)
    
    ax.add_patch(patch)
    # Font scaling: Diamond nodes (U) get smaller text to fit inside the polygon
    fs = 14 if shape != 'circle' else 24
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, fontweight='bold', color=color, zorder=11)
    
    if sublabel:
        # Increase gap to 0.7 to ensure no overlap with arrows/nodes
        if sub_pos == 'below':
            sy = y - (radius + 0.6)
            va = 'top'
        else:
            sy = y + (radius + 0.6)
            va = 'bottom'
        
        ax.text(x, sy, sublabel, ha='center', va=va, fontsize=11, 
                color=C_NAVY_MEDIUM, fontweight='bold', zorder=11,
                bbox=dict(facecolor='white', alpha=0.8, lw=0, pad=1))

def draw_interv_box(ax, x, y, label, sublabel='', w=2.8, h=1.2, color=C_CRIMSON, bg=C_PEACH):
    glow = mpatches.FancyBboxPatch((x-w/2+0.05, y-h/2-0.05), w, h, boxstyle="round,pad=0.05", 
                                   facecolor='#FEE2E2', alpha=0.6, zorder=9)
    ax.add_patch(glow)
    
    rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.05", 
                                   linewidth=2.5, edgecolor=color, facecolor=bg, zorder=10)
    ax.add_patch(rect)
    # Adjusted text positioning inside box
    ax.text(x, y + 0.15, label, ha='center', va='center', fontsize=20, fontweight='bold', color=color, zorder=11)
    if sublabel:
        ax.text(x, y - 0.35, sublabel, ha='center', va='center', fontsize=10, 
                color=C_SLATE, fontweight='bold', zorder=11)

def draw_flow(ax, x1, y1, x2, y2, color=C_NAVY_DEEP, label='', dashed=False, lw=2.2, shrink=24, connectionstyle='arc3,rad=0.0', label_pos='auto'):
    style = '--' if dashed else '-'
    arrow = ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=style, 
                        shrinkA=shrink, shrinkB=shrink, mutation_scale=22, 
                        connectionstyle=connectionstyle), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        if label_pos == 'auto':
            if connectionstyle != 'arc3,rad=0.0':
                my -= 0.8 # Move below for the check curve
            else:
                my += 0.5 # Move above for horizontal flows
        else:
            my = label_pos
             
        ax.text(mx, my, label, fontsize=15, color=color, 
                fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.95, edgecolor='none', pad=1), zorder=12)

# --- Background Zones (Expanded) -------------------------
rect_a = mpatches.FancyBboxPatch((0.1, 8.5), 11.8, 6.5, boxstyle="round,pad=0.1", 
                                 facecolor=C_MINT, edgecolor='#10B981', lw=1.5, alpha=0.15, zorder=1)
ax.add_patch(rect_a)

rect_b = mpatches.FancyBboxPatch((0.1, -1.0), 11.8, 8.8, boxstyle="round,pad=0.1", 
                                 facecolor=C_PEACH, edgecolor='#EF4444', lw=1.5, alpha=0.15, zorder=1)
ax.add_patch(rect_b)

# --- PART (A): Authentic Causal Model (Consistent Physics) ---
Y_A = 10.5
# Title moved significantly up
ax.text(0.4, 14.5, r"(A) Stable Physical Mechanism (Authentic)", 
        fontsize=22, fontweight='bold', color=C_EMERALD, zorder=20)

# Nodes - Vertical expansion
draw_node(ax, 3.5, 13.0, r'$U_{PA}$', sublabel='Exogenous Context', shape='diamond', radius=0.3, sub_pos='above')
draw_node(ax, 3.5, 10.5, r'$\mathbf{PA}_i$', sublabel='Spatio-Physical Neighbors', color=C_EMERALD)
draw_node(ax, 8.0, 13.0, r'$U_i$', sublabel='Sensor Noise', shape='diamond', radius=0.3, sub_pos='above')
draw_node(ax, 8.0, 10.5, r'$\Phi_i$', sublabel='Consistent Physical State', color=C_EMERALD)

# Flows
draw_flow(ax, 3.5, 13.0, 3.5, 10.5, shrink=12)
draw_flow(ax, 8.0, 13.0, 8.0, 10.5, shrink=12, label=r'$\sigma$')
draw_flow(ax, 3.5, 10.5, 8.0, 10.5, label=r'$f(\cdot, \kappa)$', lw=3.0, color=C_EMERALD)

# Consistent indicator
ax.text(10.5, 11.0, r'$\checkmark$', fontsize=55, color=C_EMERALD, fontweight='bold', ha='center', va='center')
ax.text(10.5, 9.8, 'Consistent', fontsize=12, color=C_EMERALD, fontweight='bold', ha='center')

# --- PART (B): Forgery Intervention (Causal Break) ------------
Y_B = 2.5
# Title moved up to clear Part B contents
ax.text(0.4, 7.3, r"(B) Forgery Intervention & Causal Violation", 
        fontsize=22, fontweight='bold', color=C_CRIMSON, zorder=20)

# Nodes
draw_node(ax, 3.5, 5.0, r'$U_{PA}$', shape='diamond', radius=0.3, sub_pos='above')
draw_node(ax, 3.5, 2.5, r'$\mathbf{PA}_i$', sublabel='Spatio-Physical Neighbors')
draw_node(ax, 8.0, 5.0, r'$U_i$', shape='diamond', radius=0.3, sub_pos='above')
draw_node(ax, 8.0, 2.5, r'$\Phi_i^*$', color=C_CRIMSON, bg='#FFF5F5', sublabel=r'Intervened State ($\Phi^*$)')
draw_node(ax, 11.0, 2.5, r'$S_i$', color=C_AMBER, bg='#FFFBEB', sublabel='Anomaly Score')

# Intervention Logic - Repositioned to avoid nodes/labels
draw_interv_box(ax, 5.75, 5.8, r'$do(\Phi_i = \Phi^*)$', sublabel='Exogenous Local Forgery', 
                color=C_CRIMSON, bg='#FFE4E6')

# Flows
draw_flow(ax, 3.5, 5.0, 3.5, 2.5, shrink=12)
draw_flow(ax, 8.0, 5.0, 8.0, 2.5, shrink=12, label=r'$\sigma$')

# THE SEVERED LINK
draw_flow(ax, 3.5, 2.5, 8.0, 2.5, color='#94A3B8', dashed=True, lw=2.5)
ax.text(5.75, 2.5, r'$\mathbf{\times}$', ha='center', va='center', fontsize=55, color=C_CRIMSON, fontweight='bold', zorder=15)
ax.text(5.75, 1.8, 'Causal Link Severed', ha='center', fontsize=12, color=C_CRIMSON, fontweight='bold', fontstyle='italic')

# Intervention forcing
draw_flow(ax, 5.75, 5.2, 8.0, 3.1, color=C_CRIMSON, lw=3.0, label=r'$do()$')

# Verification Logic (Consistency Check)
# Curved arrow at bottom - moved labels to avoid text clashing
draw_flow(ax, 3.5, 2.5, 11.0, 2.5, color=C_GOLD, lw=1.8, dashed=True, connectionstyle='arc3,rad=-0.45', shrink=25)
draw_flow(ax, 8.0, 2.5, 11.0, 2.5, label='Check', color=C_AMBER, lw=2.5, label_pos=1.5)
ax.text(11.0, 1.6, r'$||\Phi_i^* - \hat{\Phi}_i|| > \min$', ha='center', va='top', fontsize=12, color=C_AMBER, fontweight='bold')

# --- Structural Assumptions (Bottom) -------------------------------
ax.text(0.5, -2.5, r"$\mathbf{Structural\ Assumptions:}$", fontsize=13, fontweight='bold', color=C_NAVY_DEEP)
ax.text(0.5, -3.1, r"1. $\kappa$-Lipschitz Lighting ($L \in \mathcal{C}$)", fontsize=11, color=C_SLATE)
ax.text(4.2, -3.1, r"2. Independent Noise ($\sigma^2 \leq \text{const}$)", fontsize=11, color=C_SLATE)
ax.text(8.2, -3.1, r"3. Intervention Detectability ($\Phi^* \notin f(\mathbf{PA}_i)$)", fontsize=11, color=C_SLATE)

# --- Save Final Outputs --------------------------------------
save_dir = './figs'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

png_out = os.path.join(save_dir, 'causal_graph_dag.png')
pdf_out = os.path.join(save_dir, 'causal_graph_dag.pdf')

plt.savefig(png_out, dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.savefig(pdf_out, bbox_inches='tight', pad_inches=0.1)
plt.close()

print(f"Final Optimized SCM Diagram Generated (Zero Overlap Guaranteed):\n -> {png_out}\n -> {pdf_out}")
