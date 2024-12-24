import matplotlib.pyplot as plt
from datasets import load_dataset
from yourbench.utils.load_task_config import _get_full_dataset_name_for_questions

def visualize_judge_results(config: dict):
    source_dataset_name = config["selected_choices"]["visualize_results"]["source_dataset_name"]
    dataset = load_dataset(_get_full_dataset_name_for_questions(config, source_dataset_name), split="train")

    print(dataset)
    
    # Initialize counters for scenarios
    scenario_wins = {}
    
    # Count wins for each scenario
    for item in dataset:
        winning_scenario = None
        if item['evaluation_result'] == 'Answer A':
            winning_scenario = eval(item['answer_a_scenario'])  # Convert string tuple to actual tuple
        elif item['evaluation_result'] == 'Answer B':
            winning_scenario = eval(item['answer_b_scenario'])  # Convert string tuple to actual tuple
            
        if winning_scenario:
            scenario_wins[winning_scenario] = scenario_wins.get(winning_scenario, 0) + 1
    
    # Get all unique scenarios to analyze commonalities
    scenarios = list(scenario_wins.keys())
    if len(scenarios) == 2:  # We expect 2 scenarios being compared
        model1, setting1 = scenarios[0]
        model2, setting2 = scenarios[1]
        
        # Determine what's common and create appropriate labels and title
        if model1 == model2:
            title = f'Win Distribution for {model1}'
            labels = [f"{setting}\n({scenario_wins[scenario]} wins)" 
                     for scenario, setting in zip(scenarios, [setting1, setting2])]
        elif setting1 == setting2:
            title = f'Win Distribution for {setting1}'
            labels = [f"{model}\n({scenario_wins[scenario]} wins)" 
                     for scenario, model in zip(scenarios, [model1, model2])]
        else:
            labels = [f"{model}\n{setting}\n({scenario_wins[scenario]} wins)" 
                     for scenario, (model, setting) in zip(scenarios, scenarios)]
            title = 'Win Distribution by Model and Setting'
    
    # Create pie chart
    plt.figure(figsize=(6, 6), dpi=300)
    plt.pie(
        scenario_wins.values(),
        labels=labels,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title(title)
    plt.axis('equal')
    
    # Save the plot
    plt.savefig(f'{config["selected_choices"]["visualize_results"]["plots_directory"]}/judge_results_pie.png', dpi=300)
    plt.close()
