# Simple Candidate Elimination Algorithm

def is_consistent(hypothesis, sample):
    for h, s in zip(hypothesis, sample):
        if h != "?" and h != s:
            return False
    return True

# Training Data  (each row: attributes + target)
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Initialize Specific and General hypotheses
S = data[0][:-1]                 # first positive example
G = [['?' for _ in S]]           # most general hypothesis

for row in data:
    inputs, output = row[:-1], row[-1]

    if output == "Yes":  # Positive example
        for i in range(len(S)):
            if S[i] != inputs[i]:
                S[i] = '?'
        G = [g for g in G if is_consistent(g, inputs)]

    else:  # Negative example
        new_G = []
        for g in G:
            if is_consistent(g, inputs):
                for i in range(len(g)):
                    if g[i] == '?':
                        if S[i] != inputs[i]:
                            new = g.copy()
                            new[i] = S[i]
                            new_G.append(new)
        G = new_G

print("\nFinal Specific hypothesis (S):", S)
print("Final General hypothesis (G):", G)
