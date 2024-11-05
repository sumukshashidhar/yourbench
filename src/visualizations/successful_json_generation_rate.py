import pandas as pd
import json
import matplotlib.pyplot as plt

# Read and process data
data = []
with open('logs/generation_stats.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Convert to DataFrame and aggregate
df = pd.DataFrame(data)
df_agg = df.groupby('generator_model').agg({
    'total_generated_responses': 'sum',
    'total_extracted_valid_xml': 'sum',
    'total_valid_jsons': 'sum',
    'total_generated_questions': 'sum'
}).reset_index()

# Clean model names and calculate normalized rate
df_agg['generator_model'] = df_agg['generator_model'].apply(lambda x: x.split('/')[-1])
df_agg['success_rate'] = df_agg['total_valid_jsons'] / (
    df_agg['total_extracted_valid_xml'] * 
    df_agg['total_generated_responses'] * 
    df_agg['total_generated_questions']
)
df_agg['success_rate'] = df_agg['success_rate'] * 1e7  # Scale up for better visibility

# Sort and plot
plt.figure(figsize=(10, 6), dpi=300)
df_plot = df_agg.sort_values('success_rate', ascending=True)

# Create bar plot
bars = plt.barh(range(len(df_plot)), df_plot['success_rate'], color='#2ecc71')

# Customize
plt.grid(True, alpha=0.3, axis='x')
plt.xlabel('(Valid JSONs / Valid XML) / (Total Responses × Generated Questions) × 10M')
plt.ylabel('Model')
plt.title('Normalized JSON Generation Success Rate (×1M)')
plt.yticks(range(len(df_plot)), df_plot['generator_model'], fontsize=8)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, i, f'{width:.2f}', 
            va='center', ha='left', fontsize=8)

plt.tight_layout()
plt.savefig('plots/normalized_json_success_rate.png', dpi=300)
plt.close()

# Print the actual values for verification
print(df_agg[['generator_model', 'success_rate']].sort_values('success_rate', ascending=False))
