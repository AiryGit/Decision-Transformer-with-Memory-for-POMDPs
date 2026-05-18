import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys

def parse_training_log(filename):
    """Парсинг лог-файла тренировки"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Извлечение информации о датасете
    dataset_match = re.search(r'Dataset stats: total=(\d+), train=(\d+), val=(\d+), state_dim=(\d+), actions=(\d+)', content)
    if dataset_match:
        print(f"Dataset: total={dataset_match.group(1)}, train={dataset_match.group(2)}, val={dataset_match.group(3)}")
        print(f"State dim={dataset_match.group(4)}, Actions={dataset_match.group(5)}")

    # Парсинг эпох
    epochs = []
    epoch_pattern = r'Epoch (\d+)/\d+: Train Loss=([\d.]+), Val Loss=([\d.]+)'

    for match in re.finditer(epoch_pattern, content):
        epoch_num = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        epochs.append({
            'epoch': epoch_num,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

    # Парсинг валидационных результатов
    val_results = []
    val_pattern = r'Validation: Mean Return=([\d.]+), Success Rate=([\d.]+)%'

    for match in re.finditer(val_pattern, content):
        mean_return = float(match.group(1))
        success_rate = float(match.group(2))
        val_results.append({
            'mean_return': mean_return,
            'success_rate': success_rate
        })

    # Парсинг детальных результатов валидации по эпизодам
    val_episodes = []
    val_ep_pattern = r'Episode (\d+): Return=([\d.]+), Steps=(\d+)'

    for match in re.finditer(val_ep_pattern, content):
        episode = int(match.group(1))
        return_val = float(match.group(2))
        steps = int(match.group(3))
        val_episodes.append({
            'episode': episode,
            'return': return_val,
            'steps': steps
        })

    # Извлечение информации о лучшей модели
    best_match = re.search(r'Best model saved to (.+)', content)
    if best_match:
        print(f"Best model: {best_match.group(1)}")

    return epochs, val_results, val_episodes

def plot_training_curves(epochs, val_results, val_episodes, save_file=False, file_name_prefix=None):
    """График кривых обучения"""
    if not epochs:
        print("No epoch data found")
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

    # График потерь
    ax1 = axes[0, 0]
    epochs_num = [e['epoch'] for e in epochs]
    train_losses = [e['train_loss'] for e in epochs]
    val_losses = [e['val_loss'] for e in epochs]

    ax1.plot(epochs_num, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs_num, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Производная потерь (скорость изменения)
    ax2 = axes[0, 1]

    train_derivative = np.gradient(train_losses)
    val_derivative = np.gradient(val_losses)
    epochs_range = range(1, len(epochs) + 1)

    ax2.plot(epochs_range, train_derivative, 'o-', label='Train Loss Derivative', linewidth=2, markersize=6)
    ax2.plot(epochs_range, val_derivative, 's-', label='Val Loss Derivative', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss Change Rate', fontsize=12)
    ax2.set_title('Convergence Analysis - Loss Derivatives', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]

    steps = [ep['steps'] for ep in val_episodes]
    success_count = sum(1 for s in steps if s == 500)
    partial_count = sum(1 for s in steps if 400 <= s < 500)
    fail_count = sum(1 for s in steps if s < 400)

    categories = ['Full Success\n(500 steps)', 'Partial\n(400-499 steps)', 'Fail\n(<400 steps)']
    counts = [success_count, partial_count, fail_count]
    colors = ['green', 'orange', 'red']
    bars = ax3.bar(categories, counts, color=colors, edgecolor='black', alpha=0.7)
    ax3.set_ylabel('Number of Episodes', fontsize=12)
    ax3.set_title('Episode Outcomes', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    ax3.grid(True, alpha=0.3, axis='y')

    # График вознаграждений
    ax4 = axes[1, 1]
    epochs_num = [e['epoch'] for e in epochs]
    if val_results:
        returns = [r['mean_return'] for r in val_results]
        success_rates = [r['success_rate'] for r in val_results]

        ax4.plot(epochs_num[:len(returns)], returns, 'o-', label='Mean Return', linewidth=2, markersize=8, color='green')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(epochs_num[:len(success_rates)], success_rates, 's-', label='Success Rate (%)',
                     linewidth=2, markersize=8, color='orange')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Mean Return', fontsize=12, color='green')
        ax4_twin.set_ylabel('Success Rate (%)', fontsize=12, color='orange')
        ax4.tick_params(axis='y', labelcolor='green')
        ax4_twin.tick_params(axis='y', labelcolor='orange')
        ax4.set_title('Validation Performance', fontsize=14, fontweight='bold')

        # Добавление легенды
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        ax4.grid(True, alpha=0.3)

    # Гистограмма шагов
    ax5 = axes[2, 0]
    steps = [ep['steps'] for ep in val_episodes]
    bins = np.arange(0, max(steps) + 50, 50)
    ax5.hist(steps, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.set_xlabel('Steps per Episode', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Distribution of Episode Lengths', fontsize=14, fontweight='bold')
    ax5.axvline(x=np.mean(steps), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(steps):.1f}')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Шаги по эпизодам
    ax6 = axes[2, 1]
    episodes_num = [ep['episode'] for ep in val_episodes]
    steps_values = [ep['steps'] for ep in val_episodes]

    colors = ['green' if s == 500 else 'orange' if s > 400 else 'red' for s in steps_values]
    ax6.scatter(episodes_num, steps_values, c=colors, alpha=0.6, s=50)
    ax6.plot(episodes_num, steps_values, 'k-', alpha=0.3, linewidth=1)
    ax6.set_xlabel('Episode', fontsize=12)
    ax6.set_ylabel('Steps', fontsize=12)
    ax6.set_title('Episode Lengths Over Time', fontsize=14, fontweight='bold')
    ax6.axhline(y=500, color='green', linestyle='--', linewidth=2, label='Maximum (500)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_file:    
        plt.savefig(file_name_prefix + '_' + 'training_curves_analysis.png', dpi=300, bbox_inches='tight')
        print("Графики сохранены в файл:", file_name_prefix + '_' +'training_curves_analysis.png')
    plt.show()

def print_results(epochs, val_results, val_episodes):
    print(f"\nИзвлечено {len(epochs)} эпох")
    print(f"Извлечено {len(val_episodes)} валидационных эпизодов")
    print(f"Финальная тренировочная потеря: {epochs[-1]['train_loss']:.4f}")
    print(f"Финальная валидационная потеря: {epochs[-1]['val_loss']:.4f}")
    if val_results:
        print(f"Финальное среднее вознаграждение: {val_results[-1]['mean_return']:.2f}")
        print(f"Финальный процент успеха: {val_results[-1]['success_rate']:.1f}%")

# Основная функция
def main():
    # Парсинг файла
    epochs, val_results, val_episodes = parse_training_log(sys.argv[1])

    if not epochs:
        print("Не удалось извлечь данные из файла. Проверьте формат лога.")
        return
    
    # Вывод результатов
    print_results(epochs, val_results, val_episodes)

    # Построение всех графиков
    plot_training_curves(epochs, val_results, val_episodes, save_file=True, file_name_prefix = str(sys.argv[1])[:-4])



if __name__ == "__main__":
    main()