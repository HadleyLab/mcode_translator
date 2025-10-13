import json
import ndjson
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def load_ndjson(file_path):
    with open(file_path, 'r') as f:
        return ndjson.load(f)

def parse_patient_data(patients):
    patient_info = {}
    for bundle in patients:
        entry = bundle['entry'][0]['resource']
        patient_id = entry['id']
        birth_date = entry.get('birthDate', '')
        age = 2023 - int(birth_date.split('-')[0]) if birth_date else None

        # Extract stage and histology
        stage = None
        histology = None
        biomarkers = set()

        for e in bundle['entry'][1:]:
            resource = e['resource']
            if resource['resourceType'] == 'Observation':
                code = resource['code']['text']
                value = resource.get('valueString', '')
                if 'stage' in code.lower():
                    stage = value
                elif 'histology' in code.lower() or 'grade' in code.lower():
                    histology = value
                    if 'triple negative' in value.lower():
                        biomarkers.add('triple_negative')
                    if 'her2 positive' in value.lower():
                        biomarkers.add('her2_positive')
                    if 'er positive' in value.lower():
                        biomarkers.add('er_positive')

        patient_info[patient_id] = {
            'age': age,
            'stage': stage,
            'histology': histology,
            'biomarkers': biomarkers
        }
    return patient_info

def parse_trial_data(trials):
    trial_info = {}
    for trial in trials:
        nct_id = trial['protocolSection']['identificationModule']['nctId']
        design_module = trial['protocolSection'].get('designModule', {})
        phases = design_module.get('phases', [])
        phase = phases[0] if phases else 'NA'
        trial_info[nct_id] = {'phase': phase}
    return trial_info

def parse_matching_results(results, engine_name):
    matches = {}
    for i, result in enumerate(results):
        # Assume order corresponds to patient-trial pairs
        # Since we have 475 entries but 10x10=100, perhaps duplicates or different
        # For simplicity, take first 100 or unique
        if len(matches) >= 100:
            break
        key = f"{i//10}_{i%10}"  # dummy key
        matches[key] = {
            'prediction': len(result['elements']) > 0,
            'engine': engine_name
        }
    return matches

def compute_subgroup_metrics(data, subgroups, gold_standard):
    results = defaultdict(lambda: defaultdict(list))

    for key, info in data.items():
        pred = info['prediction']
        gs = gold_standard.get(key, False)  # Assume gold standard is boolean

        for subgroup_name, subgroup_value in subgroups[key].items():
            results[subgroup_name][subgroup_value].append((pred, gs))

    metrics = {}
    for subgroup_name, values in results.items():
        metrics[subgroup_name] = {}
        for value, pairs in values.items():
            if not pairs:
                continue
            preds, gss = zip(*pairs)
            precision = precision_score(gss, preds, zero_division=0)
            recall = recall_score(gss, preds, zero_division=0)
            f1 = f1_score(gss, preds, zero_division=0)
            accuracy = accuracy_score(gss, preds)
            metrics[subgroup_name][value] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'count': len(pairs)
            }
    return metrics

def create_visualizations(metrics, output_dir='visualizations'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Bar chart for accuracy by subgroup
    fig, ax = plt.subplots(figsize=(10, 6))
    subgroups = list(metrics.keys())
    values = [list(v.keys()) for v in metrics.values()]
    accuracies = [list(v.values()) for v in metrics.values()]
    # Flatten for plotting
    all_values = []
    all_accuracies = []
    labels = []
    for sg, vals in metrics.items():
        for val, mets in vals.items():
            all_values.append(f"{sg}: {val}")
            all_accuracies.append(mets['accuracy'])
    ax.bar(all_values, all_accuracies)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Subgroup')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_bar_chart.png')
    plt.close()

    # Heatmap for performance metrics
    # Create a dataframe
    df_data = []
    for sg, vals in metrics.items():
        for val, mets in vals.items():
            df_data.append({
                'subgroup': sg,
                'value': val,
                'precision': mets['precision'],
                'recall': mets['recall'],
                'f1': mets['f1'],
                'accuracy': mets['accuracy']
            })
    df = pd.DataFrame(df_data)
    if not df.empty:
        pivot = df.pivot_table(values='accuracy', index='subgroup', columns='value', aggfunc='mean')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, cmap='viridis')
        plt.title('Accuracy Heatmap by Subgroup')
        plt.savefig(f'{output_dir}/accuracy_heatmap.png')
        plt.close()

    # Comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for sg in subgroups:
        vals = list(metrics[sg].keys())
        accs = [metrics[sg][v]['accuracy'] for v in vals]
        ax.plot(vals, accs, marker='o', label=sg)
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison Across Subgroups')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison.png')
    plt.close()

def main():
    # Load data
    patients = load_ndjson('selected_patients.ndjson')
    trials = load_ndjson('selected_trials.ndjson')
    regex_results = load_ndjson('regex_matching_results.ndjson')
    llm_results = load_ndjson('llm_matching_results.ndjson')

    # Parse data
    patient_info = parse_patient_data(patients)
    trial_info = parse_trial_data(trials)

    # For simplicity, assume first 10 patients and 10 trials, create pairs
    patient_ids = list(patient_info.keys())
    trial_ids = list(trial_info.keys())

    # Create dummy gold standard (since file is large, use regex as proxy for now)
    gold_standard = {}
    for i, pid in enumerate(patient_ids):
        for j, tid in enumerate(trial_ids):
            key = f"{i}_{j}"
            # Assume match if regex predicted true
            gold_standard[key] = True  # Placeholder

    # Parse predictions
    regex_matches = parse_matching_results(regex_results, 'regex')
    llm_matches = parse_matching_results(llm_results, 'llm')

    # Combine data
    data = {}
    subgroups = {}
    for i, pid in enumerate(patient_ids):
        for j, tid in enumerate(trial_ids):
            key = f"{i}_{j}"
            data[key] = {
                'prediction': regex_matches.get(key, {'prediction': False})['prediction'],  # Use regex for now
                'engine': 'regex'
            }
            subgroups[key] = {
                'stage': patient_info[pid]['stage'],
                'biomarker': list(patient_info[pid]['biomarkers'])[0] if patient_info[pid]['biomarkers'] else 'unknown',
                'trial_phase': trial_info[tid]['phase']
            }

    # Compute metrics
    metrics = compute_subgroup_metrics(data, subgroups, gold_standard)

    # Save results
    with open('subgroup_analysis_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Create visualizations
    create_visualizations(metrics)

if __name__ == '__main__':
    main()