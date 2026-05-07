import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

t = np.linspace(0, 1, 100)

# Base curve (learning reward pattern)
base = 0.45 + 0.9*np.sin(np.pi*t) + 0.25*np.sin(3*np.pi*t)

fig, axes = plt.subplots(4,4, figsize=(15,11))

titles = [
    "Original Functions",
    "Shape PCA - Amplitude",
    "Shape PCA - Phase",
    "fPCA"
]


for r in range(4):
    for c in range(4):

        ax = axes[r,c]

        for i in range(18):

            amp = 1 + np.random.uniform(-0.25,0.25)
            shift = np.random.uniform(-0.05,0.05)

            x2 = np.clip(t + shift,0,1)

            y = amp*(0.45 + 0.9*np.sin(np.pi*x2) + 0.25*np.sin(3*np.pi*x2))

            noise = np.random.normal(0,0.02,len(y))
            y = y + noise

            ax.plot(t,y,lw=1)

        ax.set_xlim(0,1)
        ax.grid(alpha=0.2)

        if r == 0:
            ax.set_title(titles[c], fontsize=10)

# Left labels
axes[0,0].text(-0.35,1.9,"(A)",fontsize=13,fontweight='bold')
axes[1,0].text(-0.35,1.9,"(B)",fontsize=13,fontweight='bold')
axes[2,0].text(-0.35,1.9,"(C)",fontsize=13,fontweight='bold')
axes[3,0].text(-0.35,1.9,"(D)",fontsize=13,fontweight='bold')

plt.tight_layout()
plt.savefig("metrics_fpca_style.png", dpi=400)
plt.show()