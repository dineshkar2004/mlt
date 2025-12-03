import numpy as np

data = np.array([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
])

concepts = data[:, :-1]
target = data[:, -1]

def learn(concepts, target):
    specific_h = None
    general_h = [['?' for _ in range(len(concepts[0]))] for _ in range(len(concepts[0]))]

    for i, val in enumerate(target):
        if val == 'Yes':
            if specific_h is None:
                specific_h = concepts[i].copy()
            else:
                for x in range(len(specific_h)):
                    if concepts[i][x] != specific_h[x]:
                        specific_h[x] = '?'
            for x in range(len(general_h)):
                if general_h[x][x] != '?':
                    general_h[x][x] = '?'
        elif val == 'No':
            for x in range(len(concepts[0])):
                if specific_h[x] != concepts[i][x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    indices = [i for i, val in enumerate(general_h) if val != ['?', '?', '?', '?', '?', '?']]
    return specific_h, [general_h[i] for i in indices]

s_final, g_final = learn(concepts, target)

print("\nFinal Specific Hypothesis:", s_final)
print("Final General Hypotheses:", g_final)
