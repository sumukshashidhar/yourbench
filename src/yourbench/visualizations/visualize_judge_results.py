import plotly.graph_objects as go
from datasets import load_dataset

from yourbench.utils.load_task_config import _get_full_dataset_name_for_questions


def _count_questions_per_category(dataset):
    total_questions_per_category = {}
    for item in dataset:
        cat = item['question_type']
        total_questions_per_category[cat] = total_questions_per_category.get(cat, 0) + 1
    return total_questions_per_category


def _collect_scenario_wins(dataset):
    scenario_category_wins = {}
    for item in dataset:
        winning_scenario = None
        if item['evaluation_result'] == 'Answer A':
            winning_scenario = eval(item['answer_a_scenario'])
        elif item['evaluation_result'] == 'Answer B':
            winning_scenario = eval(item['answer_b_scenario'])

        if winning_scenario:
            category = item['question_type']
            scenario_category_wins.setdefault(winning_scenario, {})
            scenario_category_wins[winning_scenario][category] = \
                scenario_category_wins[winning_scenario].get(category, 0) + 1
    return scenario_category_wins


def _compute_normalized_wins(scenario_category_wins, total_questions_per_category, total_questions_in_dataset):
    total_wins_per_category = {}
    for scenario, cat_dict in scenario_category_wins.items():
        for cat, wins_count in cat_dict.items():
            total_wins_per_category[cat] = total_wins_per_category.get(cat, 0) + wins_count

    scenario_category_wins_normalized = {}
    for scenario, cat_dict in scenario_category_wins.items():
        scenario_category_wins_normalized[scenario] = {}
        for cat, scenario_wins_in_cat in cat_dict.items():
            total_wins_in_cat = total_wins_per_category[cat]
            if total_wins_in_cat == 0:
                fraction = 0
            else:
                fraction_s_share = scenario_wins_in_cat / total_wins_in_cat
                fraction_cat_weight = total_questions_per_category[cat] / total_questions_in_dataset
                fraction = fraction_s_share * fraction_cat_weight
            scenario_category_wins_normalized[scenario][cat] = fraction
    return scenario_category_wins_normalized


def _prepare_sunburst_data(scenario_category_wins_normalized):
    labels, ids, parents, values = [], [], [], []
    scenarios = list(scenario_category_wins_normalized.keys())

    if len(scenarios) == 2:
        model1, setting1 = scenarios[0]
        model2, setting2 = scenarios[1]

        if model1 == model2:
            top_labels = [setting1, setting2]
            chart_title = f'Win Distribution for {model1}'
        elif setting1 == setting2:
            top_labels = [model1, model2]
            chart_title = f'Win Distribution for {setting1}'
        else:
            top_labels = [f"{model1}-{setting1}", f"{model2}-{setting2}"]
            chart_title = 'Win Distribution by Model and Setting'

        for scenario, top_label in zip(scenarios, top_labels):
            scenario_id = f"{scenario[0]}::{scenario[1]}"
            total_fraction = sum(scenario_category_wins_normalized[scenario].values())

            labels.append(top_label)
            ids.append(scenario_id)
            parents.append("")
            values.append(total_fraction)

            for category, fraction_value in scenario_category_wins_normalized[scenario].items():
                category_id = f"{scenario_id}::{category}"
                labels.append(category)
                ids.append(category_id)
                parents.append(scenario_id)
                values.append(fraction_value)

    return labels, ids, parents, values, chart_title


def visualize_judge_results(config: dict):
    source_dataset_name = config["selected_choices"]["visualize_results"]["source_dataset_name"]
    dataset = load_dataset(_get_full_dataset_name_for_questions(config, source_dataset_name), split="train")

    total_questions_per_category = _count_questions_per_category(dataset)
    total_questions_in_dataset = sum(total_questions_per_category.values())

    scenario_category_wins = _collect_scenario_wins(dataset)
    scenario_category_wins_normalized = _compute_normalized_wins(
        scenario_category_wins, total_questions_per_category, total_questions_in_dataset
    )

    labels, ids, parents, values, chart_title = _prepare_sunburst_data(scenario_category_wins_normalized)

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
        )
    )

    fig.update_layout(
        title=chart_title,
        width=800,
        height=800,
        font={"size": 20}
    )

    plots_dir = config['selected_choices']['visualize_results']['plots_directory']
    fig.write_html(f"{plots_dir}/judge_results_sunburst.html")
    fig.write_image(f"{plots_dir}/judge_results_sunburst.png", scale=2, width=800, height=800)
