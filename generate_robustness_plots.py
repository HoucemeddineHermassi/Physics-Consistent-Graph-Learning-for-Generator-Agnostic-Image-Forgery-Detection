import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_plots():
    results_path = 'experimental_results.json'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    robust = data.get('robustness', {})
    if not robust:
        print("Error: Robustness data not found in results.")
        return

    # Create figs directory if it doesn't exist
    figs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elsarticle (1)', 'elsarticle', 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    # Plotting style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    colors = {'TruFor': '#1f77b4', 'NoisePrint++': '#ff7f0e', 'VLM-Reasoning': '#2ca02c', 'CPCL (Ours)': '#d62728'}
    markers = {'TruFor': 'o', 'NoisePrint++': 's', 'VLM-Reasoning': 'D', 'CPCL (Ours)': '*'}
    
    def plot_line(ax_or_plt, x, values, model, is_pcgl):
        label = 'PCGL (Ours)' if is_pcgl else model
        lw = 3 if is_pcgl else 2
        ms = 10 if is_pcgl else 7
        ax_or_plt.plot(x, values, label=label, color=colors[model], marker=markers[model], linewidth=lw, markersize=ms)

    # 1. JPEG Robustness
    plt.figure(figsize=(8, 5), dpi=300)
    x = robust['JPEG']['Qualities']
    for model, values in robust['JPEG']['Models'].items():
        plot_line(plt, x, values, model, 'CPCL' in model)
    plt.xlabel('JPEG Quality Factor', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=14, fontweight='bold')
    plt.title('Performance under JPEG Compression', fontsize=16, fontweight='bold', pad=10)
    plt.ylim(0, 1.0)
    plt.gca().invert_xaxis()
    plt.legend(frameon=True, fontsize=12, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'robust_jpeg.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Blur Robustness
    plt.figure(figsize=(8, 5), dpi=300)
    x = robust['Blur']['Sigmas']
    for model, values in robust['Blur']['Models'].items():
        plot_line(plt, x, values, model, 'CPCL' in model)
    plt.xlabel('Gaussian Blur Sigma ($\sigma$)', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=14, fontweight='bold')
    plt.title('Performance under Gaussian Blur', fontsize=16, fontweight='bold', pad=10)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, fontsize=12, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'robust_blur.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Resizing Robustness
    plt.figure(figsize=(8, 5), dpi=300)
    x = robust['Resizing']['Factors']
    for model, values in robust['Resizing']['Models'].items():
        plot_line(plt, x, values, model, 'CPCL' in model)
    plt.xlabel('Resizing Factor (Scale)', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=14, fontweight='bold')
    plt.title('Performance under Geometric Resizing', fontsize=16, fontweight='bold', pad=10)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, fontsize=12, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'robust_resizing.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Adversarial Robustness (White vs Black)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    x = robust['Adversarial_WhiteBox']['Epsilons']
    for model, values in robust['Adversarial_WhiteBox']['Models'].items():
        plot_line(ax1, x, values, model, 'CPCL' in model)
    ax1.set_xlabel('Perturbation Magnitude ($\epsilon$)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax1.set_title('White-Box Adversarial Attack', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.legend(frameon=True, fontsize=12, shadow=True)

    for model, values in robust['Adversarial_BlackBox']['Models'].items():
        plot_line(ax2, x, values, model, 'CPCL' in model)
    ax2.set_xlabel('Perturbation Magnitude ($\epsilon$)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax2.set_title('Black-Box Adversarial Attack', fontsize=16, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.legend(frameon=True, fontsize=12, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'robust_adversarial.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Recursive Screenshot
    plt.figure(figsize=(8, 5), dpi=300)
    x = robust['Recursive_Screenshot']['Rounds']
    for model, values in robust['Recursive_Screenshot']['Models'].items():
        plot_line(plt, x, values, model, 'CPCL' in model)
    plt.xlabel('Recursive Rounds', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=14, fontweight='bold')
    plt.title('Performance under Recursive Screenshotting', fontsize=16, fontweight='bold', pad=10)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, fontsize=12, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'robust_screenshot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Robustness plots generated successfully.")

if __name__ == "__main__":
    generate_plots()
