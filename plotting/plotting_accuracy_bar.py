import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["HateBERT", "BERTweet", "BERT", "MLP"]

dev_acc = [98.79, 98.51, 97.95, 97.98]
exact_match = [92.50, 92.77, 90.62, 91.63]

x = np.arange(len(models))
width = 0.35

plt.figure()

bars1 = plt.bar(x - width/2, dev_acc, width, label="Dev Accuracy")
bars2 = plt.bar(x + width/2, exact_match, width, label="Exact Match Accuracy")

plt.xticks(x, models)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Dev Accuracy vs Exact Match Accuracy by Model")
plt.legend()
plt.grid(True)
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.1,
             f"{height:.2f}%", ha="center", va="bottom", fontsize=8)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.1,
             f"{height:.2f}%", ha="center", va="bottom", fontsize=8)
plt.show()