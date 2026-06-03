# src/p1_dataset/calculate_kappa.py
# Run this after all 3 annotators have filled their labels

from sklearn.metrics import cohen_kappa_score
import pandas as pd

# Each person saves their annotations as a separate CSV
# Load them here:
p1 = pd.read_csv('data/annotated/labels_person1.csv')['greenwash_label']
p2 = pd.read_csv('data/annotated/labels_person2.csv')['greenwash_label']
p3 = pd.read_csv('data/annotated/labels_person3.csv')['greenwash_label']

# Convert to numbers
label_map = {'Low': 0, 'Medium': 1, 'High': 2}

p1n = p1.map(label_map)
p2n = p2.map(label_map)
p3n = p3.map(label_map)

kappa_12 = cohen_kappa_score(p1n, p2n)
kappa_13 = cohen_kappa_score(p1n, p3n)
kappa_23 = cohen_kappa_score(p2n, p3n)

print(f'Kappa P1 vs P2: {kappa_12:.3f}')
print(f'Kappa P1 vs P3: {kappa_13:.3f}')
print(f'Kappa P2 vs P3: {kappa_23:.3f}')
print(f'Average Kappa:  {(kappa_12+kappa_13+kappa_23)/3:.3f}')
print('Target: >= 0.70')

# For disagreements, use majority vote
final_df = pd.DataFrame({'p1': p1, 'p2': p2, 'p3': p3})
final_df['final_label'] = final_df.mode(axis=1)[0]
final_df.to_csv('data/annotated/final_labels.csv', index=False)
print('Final labels saved to data/annotated/final_labels.csv')
