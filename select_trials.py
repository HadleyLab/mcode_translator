import json
import random

def load_trials(filename):
    trials = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    return trials

def extract_trial_info(trial):
    protocol = trial.get('protocolSection', {})
    ident = protocol.get('identificationModule', {})
    status = protocol.get('statusModule', {})
    design = protocol.get('designModule', {})
    arms = protocol.get('armsInterventionsModule', {})

    nct_id = ident.get('nctId', '')
    title = ident.get('briefTitle', '')
    overall_status = status.get('overallStatus', '')
    phases = design.get('phases', [])
    interventions = arms.get('interventions', [])

    # Extract phase
    phase = 'Not Applicable'
    if phases:
        phase = ', '.join(phases)

    # Extract intervention types
    intervention_types = set()
    for interv in interventions:
        itype = interv.get('type', '')
        if itype:
            intervention_types.add(itype)

    intervention_types = ', '.join(sorted(intervention_types))

    return {
        'nct_id': nct_id,
        'title': title,
        'status': overall_status,
        'phase': phase,
        'intervention_types': intervention_types,
        'trial': trial
    }

def select_diverse_trials(trials, n=10):
    infos = [extract_trial_info(t) for t in trials]

    # Group by phase
    by_phase = {}
    for info in infos:
        phase = info['phase']
        if phase not in by_phase:
            by_phase[phase] = []
        by_phase[phase].append(info)

    # Group by intervention types
    by_intervention = {}
    for info in infos:
        interv = info['intervention_types']
        if interv not in by_intervention:
            by_intervention[interv] = []
        by_intervention[interv].append(info)

    selected = []
    phases_used = set()
    interventions_used = set()

    # First, try to get one from each phase
    for phase, phase_trials in by_phase.items():
        if phase_trials:
            trial = random.choice(phase_trials)
            selected.append(trial)
            phases_used.add(phase)
            interventions_used.add(trial['intervention_types'])
            if len(selected) >= n:
                break

    # If still need more, get from different intervention types
    if len(selected) < n:
        for interv, interv_trials in by_intervention.items():
            if interv not in interventions_used:
                for trial in interv_trials:
                    if trial not in selected:
                        selected.append(trial)
                        interventions_used.add(interv)
                        if len(selected) >= n:
                            break
                if len(selected) >= n:
                    break

    # If still need more, just add random
    remaining = [t for t in infos if t not in selected]
    while len(selected) < n and remaining:
        trial = random.choice(remaining)
        selected.append(trial)
        remaining.remove(trial)

    return selected[:n]

def main():
    trials = load_trials('fetched_trials.ndjson')
    print(f"Loaded {len(trials)} trials")

    selected = select_diverse_trials(trials, 10)

    print(f"Selected {len(selected)} diverse trials:")

    for i, info in enumerate(selected, 1):
        print(f"{i}. {info['nct_id']} - Phase: {info['phase']} - Interventions: {info['intervention_types']}")
        print(f"   Title: {info['title'][:100]}...")
        print()

    # Save selected trials
    with open('selected_trials.ndjson', 'w') as f:
        for info in selected:
            json.dump(info['trial'], f, ensure_ascii=False)
            f.write('\n')

    print("Saved selected trials to selected_trials.ndjson")

    # Diversity summary
    phases = set(info['phase'] for info in selected)
    interventions = set(info['intervention_types'] for info in selected)

    print("Diversity Summary:")
    print(f"- Phases represented: {sorted(phases)}")
    print(f"- Intervention types represented: {sorted(interventions)}")

if __name__ == '__main__':
    main()