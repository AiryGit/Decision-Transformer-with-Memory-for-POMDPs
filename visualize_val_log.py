import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys

def parse_log_file(filename):
    """Парсинг лог-файла для извлечения данных"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Извлечение информации о модели
    model_info = re.search(r'n_embed=(\d+), n_layer=(\d+), n_head=(\d+)', content)
    if model_info:
        print(f"Model: n_embed={model_info.group(1)}, n_layer={model_info.group(2)}, n_head={model_info.group(3)}")

    # Извлечение среднего вознаграждения
    avg_reward = re.search(r'Average reward over \d+ episodes: ([\d.]+)', content)
    if avg_reward:
        print(f"Average reward: {avg_reward.group(1)}")

    # Парсинг каждого эпизода
    episodes = []
    episode_pattern = r'Episode (\d+)/\d+\n=+\nTarget return: ([\d.]+)\n(.*?)Episode \d+ finished with reward: ([\d.]+)'

    for match in re.finditer(episode_pattern, content, re.DOTALL):
        episode_num = int(match.group(1))
        target_return = float(match.group(2))
        episode_content = match.group(3)
        reward = float(match.group(4))

        # Парсинг шагов эпизода
        steps = []
        for step_match in re.finditer(r'Step (\d+): Action=(\d+), RTG=([\d.]+)', episode_content):
            step_num = int(step_match.group(1))
            action = int(step_match.group(2))
            rtg = float(step_match.group(3))
            steps.append({'step': step_num, 'action': action, 'rtg': rtg})

        episodes.append({
            'episode': episode_num,
            'target_return': target_return,
            'reward': reward,
            'steps': steps
        })

    return episodes


def plot_validation_curves(episodes, save_file=False, file_name_prefix = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1. RTG decay (среднее по всем эпизодам)
    ax1 = axes[0]
    max_steps = max(len(ep['steps']) for ep in episodes)
    all_rtgs = []
    for episode in episodes:
        rtgs = [step['rtg'] for step in episode['steps']]
        # Дополнение до одинаковой длины
        rtgs.extend([rtgs[-1]] * (max_steps - len(rtgs)))
        all_rtgs.append(rtgs)

    mean_rtgs = np.mean(all_rtgs, axis=0)
    std_rtgs = np.std(all_rtgs, axis=0)

    steps_range = range(1, max_steps + 1)
    ax1.plot(steps_range, mean_rtgs, 'b-', linewidth=2, label='Mean RTG')
    ax1.fill_between(steps_range, mean_rtgs - std_rtgs, mean_rtgs + std_rtgs,
                     alpha=0.2, color='blue', label='±1 STD')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('RTG', fontsize=12)
    ax1.set_title('Mean RTG Decay (All Episodes)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    # 3. Вознаграждения по эпизодам
    ax2 = axes[1]
    rewards = [ep['reward'] for ep in episodes]
    ep_numbers = [ep['episode'] for ep in episodes]
    bars = ax2.bar(ep_numbers, rewards, color='gold', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax2.set_xticks(ep_numbers)
    ax2.set_ylim(0, max(rewards) * 1.1)
    ax2.grid(True, alpha=0.3, axis='y')

    # Добавление значений на столбцы
    for bar, reward in zip(bars, rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{reward:.0f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_file:    
        plt.savefig(file_name_prefix + '_' + 'validation_curves_analysis.png', dpi=300, bbox_inches='tight')
        print("Графики сохранены в файл:", file_name_prefix + '_' +'validation_curves_analysis.png')
    plt.show()

def print_results(episodes):
    print(f"\nИзвлечено {len(episodes)} эпизодов")
    print(f"Среднее вознаграждение: {np.mean([ep['reward'] for ep in episodes]):.2f}")
    print(f"Общее количество шагов: {sum(len(ep['steps']) for ep in episodes)}")

# Основная функция
def main():
    # Парсинг файла
    episodes = parse_log_file(sys.argv[1])

    if not episodes:
        print("Не удалось извлечь данные из файла. Проверьте формат лога.")
        return

    # Вывод результатов
    print_results(episodes)

    # Построение всех графиков
    plot_validation_curves(episodes, save_file=True, file_name_prefix = str(sys.argv[1])[:-4])

if __name__ == "__main__":
    main()