import matplotlib.pyplot as plt

# Plots for each model val loss over epochs
epochs = [1, 2, 3]
loss_bertweet = [0.0429, 0.0404, 0.0373]
loss_hatebert = [0.0421, 0.0394, 0.0380]
loss_mlp = [0.0616, 0.0607, 0.0599]
loss_bert = [0.0540, 0.0516, 0.0559]

plt.figure()
plt.plot(epochs, loss_hatebert, label="HateBERT")
plt.plot(epochs, loss_bertweet, label="BERTweet")
plt.plot(epochs, loss_bert, label="BERT")
plt.plot(epochs, loss_mlp, label="MLP")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss All Models over Epochs")
plt.xticks([1, 2, 3])
plt.legend()
plt.grid(True)
plt.show()

# Bar graph for hatebert per-class F1 scores
labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
f1_scores = [0.8202, 0.5185, 0.8338, 0.5682, 0.7806, 0.6113]

plt.figure()
bars = plt.bar(labels, f1_scores)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom"
    )

plt.bar(labels, f1_scores)
plt.xlabel("Class")
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Scores of HateBERT (Multi-Label)")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Bar graph for bert per-class F1 scores
labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
f1_scores = [0.8152, 0.5232, 0.8059, 0.5432, 0.7622, 0.5844]

plt.figure()
bars = plt.bar(labels, f1_scores)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom"
    )

plt.bar(labels, f1_scores)
plt.xlabel("Class")
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Scores of BERT (Multi-Label)")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Bar graph for mlp per-class F1 scores
labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
f1_scores = [0.7138, 0.2250, 0.7570, 0.3902, 0.6739, 0.3081]

plt.figure()
bars = plt.bar(labels, f1_scores)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom"
    )

plt.bar(labels, f1_scores)
plt.xlabel("Class")
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Scores of Classical MLP (Multi-Label)")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Bar graph for bertweet per-class F1 scores
labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
f1_scores = [0.8409, 0.3484, 0.8342, 0.5419, 0.7732, 0.5663]

plt.figure()
bars = plt.bar(labels, f1_scores)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom"
    )

plt.bar(labels, f1_scores)
plt.xlabel("Class")
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Scores of Classical BERTweet (Multi-Label)")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
