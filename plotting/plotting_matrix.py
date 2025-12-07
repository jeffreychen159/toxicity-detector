import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

## Plotting a confusion matrix for HateBERT
# Confusion matrix values
tn, fp, fn, tp = 7709, 128, 906, 1257
cm = np.array([[tp, fp],
               [fn, tn]])

cell_labels = np.array([
    ["TP", "FP"],
    ["FN", "TN"]
])

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title("Confusion Matrix HateBERT (Binary Classification)")

for i in range(2):
    for j in range(2):
        if (i == 0 and j == 0) or (i == 1 and j == 1):
            color = "green"
        else:
            color = "red"

        ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.6))
        # ax.text(j + 0.5, i + 0.35, cell_labels[i, j], ha="center", va="center", fontsize=12, color="black")
        ax.text(j + 0.5, i + 0.5, cm[i, j],ha="center", va="center", fontsize=12, color="black")

# Axis formatting
ax.set_xlim(0, 2)
ax.set_ylim(2, 0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()


## Plotting a confusion matrix for BERTweet
# Confusion matrix values
tn, fp, fn, tp = 7705, 132, 975, 1188
cm = np.array([[tp, fp],
               [fn, tn]])

cell_labels = np.array([
    ["TP", "FP"],
    ["FN", "TN"]
])

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title("Confusion Matrix BERTweet (Binary Classification)")

for i in range(2):
    for j in range(2):
        if (i == 0 and j == 0) or (i == 1 and j == 1):
            color = "green"
        else:
            color = "red"

        ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.6))
        # ax.text(j + 0.5, i + 0.35, cell_labels[i, j], ha="center", va="center", fontsize=12, color="black")
        ax.text(j + 0.5, i + 0.5, cm[i, j],ha="center", va="center", fontsize=12, color="black")

# Axis formatting
ax.set_xlim(0, 2)
ax.set_ylim(2, 0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

## Plotting a confusion matrix for MLP
# Confusion matrix values
tn, fp, fn, tp = 7167, 670, 1908, 255
cm = np.array([[tp, fp],
               [fn, tn]])

cell_labels = np.array([
    ["TP", "FP"],
    ["FN", "TN"]
])

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title("Confusion Matrix MLP (Binary Classification)")

for i in range(2):
    for j in range(2):
        if (i == 0 and j == 0) or (i == 1 and j == 1):
            color = "green"
        else:
            color = "red"

        ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.6))
        # ax.text(j + 0.5, i + 0.35, cell_labels[i, j], ha="center", va="center", fontsize=12, color="black")
        ax.text(j + 0.5, i + 0.5, cm[i, j],ha="center", va="center", fontsize=12, color="black")

# Axis formatting
ax.set_xlim(0, 2)
ax.set_ylim(2, 0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

## Plotting a confusion matrix for bert
# Confusion matrix values
tn, fp, fn, tp = 7709, 128, 906, 1257
cm = np.array([[tp, fp],
               [fn, tn]])

cell_labels = np.array([
    ["TP", "FP"],
    ["FN", "TN"]
])

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title("Confusion Matrix BERT (Binary Classification)")

for i in range(2):
    for j in range(2):
        if (i == 0 and j == 0) or (i == 1 and j == 1):
            color = "green"
        else:
            color = "red"

        ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.6))
        # ax.text(j + 0.5, i + 0.35, cell_labels[i, j], ha="center", va="center", fontsize=12, color="black")
        ax.text(j + 0.5, i + 0.5, cm[i, j],ha="center", va="center", fontsize=12, color="black")

# Axis formatting
ax.set_xlim(0, 2)
ax.set_ylim(2, 0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()