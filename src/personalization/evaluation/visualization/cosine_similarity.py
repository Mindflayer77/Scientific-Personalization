import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_similarity_heatmap(pair_df):
    """
    Expects the pair-level dataframe with columns: pair, key, mean, std, count
    'pair' is a string like "(data_engineer)" or "(interpreter, sota_chaser)"
    """
    # Parse the pair column into individual user types
    users = sorted(set(
        u for _, row in pair_df.iterrows()
        for u in list(row['pair_key'])
    ))
    matrix = pd.DataFrame(index=users, columns=users, dtype=float)

    for _, row in pair_df.iterrows():
        members = list(row['pair_key'])
        sim = row['mean']
        if len(members) == 1:
            # Intra-user → diagonal
            u = members[0]
            matrix.at[u, u] = sim
        else:
            u1, u2 = members[0], members[1]
            matrix.at[u1, u2] = sim
            matrix.at[u2, u1] = sim

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix, annot=True, cmap='YlGnBu', fmt='.2f',
        square=True, vmin=0, vmax=1,
        linewidths=0.5, linecolor='white'
    )
    plt.title('Semantic Similarity Matrix)', pad=20)
    plt.tight_layout()
    plt.show()


def plot_intra_inter_comparison(cat_df):
    """
    Expects the category-level dataframe with columns:
    user_type, category, mean, std, count
    category values: 'Intra-User (Same)' / 'Inter-User (Different)'
    """
    plot_df = cat_df.copy()
    plot_df['Comparison'] = plot_df['category'].map({
        'Intra-User (Same)':      'Intra (Self)',
        'Inter-User (Different)': 'Inter (Others)',
    })

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_df,
        x='user_type', y='mean',
        hue='Comparison',
        palette='Set2',
        capsize=0.05,
    )

    for bar, (_, row) in zip(ax.patches, plot_df.iterrows()):
        cx = bar.get_x() + bar.get_width() / 2
        cy = bar.get_height()
        ax.errorbar(cx, cy, yerr=row['std'], fmt='none', color='black', linewidth=1.2, capsize=4)

    plt.title('Intra-User vs Inter-User Semantic Similarity by User Type')
    # plt.ylim(0, max(plot_df['mean'] + plot_df['ci']) * 1.2)
    plt.ylabel('Average Cosine Similarity (mean ± std)')
    plt.xlabel('User Type')
    plt.legend(title='Comparison', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def visualize_umap(personas: list, labels, embeddings_2d, score):
    colors = plt.cm.tab10(np.linspace(0, 1, len(personas)))

    _, ax = plt.subplots(figsize=(8, 6))
    for persona, color in zip(personas, colors):
        mask = labels == persona
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                label=persona, color=color, s=80, alpha=0.85, edgecolors="white")

    ax.set_title(f"UMAP of Persona Hypotheses  |  Silhouette = {score:.3f}")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()