import json
import sys
import io

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load all comparison data
datasets = []
for i in [1, 2, 3, 4]:
    with open(f'Text Demo/RAW_data/chart_set_{i}_bcbo_comparison.json', 'r', encoding='utf-8') as f:
        datasets.append(json.load(f))

print('='*80)
print('BCBO vs BCBO-DE Performance Comparison Analysis')
print('='*80)

summary = {'bcbo_wins': 0, 'bcbode_wins': 0, 'total': 0}

for ds in datasets:
    print(f'\n{ds["config"]["name"]}')
    print('-'*80)

    bcbo_results = ds['algorithms']['BCBO']['results']
    bcbode_results = ds['algorithms']['BCBO-DE']['results']

    if 'iteration' in bcbo_results[0]:
        # Iteration-based comparison
        print(f'{"Iteration":<12} {"BCBO Fitness":<18} {"BCBO-DE Fitness":<18} {"Difference":<15} {"Winner"}')
        print('-'*80)
        for b, bd in zip(bcbo_results[::4], bcbode_results[::4]):  # Sample every 4th
            diff = bd['best_fitness'] - b['best_fitness']
            winner = 'BCBO-DE [WIN]' if diff > 0 else 'BCBO'
            print(f'{b["iteration"]:<12} {b["best_fitness"]:<18.4f} {bd["best_fitness"]:<18.4f} {diff:<15.4f} {winner}')

        # Final comparison
        final_bcbo = bcbo_results[-1]['best_fitness']
        final_bcbode = bcbode_results[-1]['best_fitness']
        final_diff = final_bcbode - final_bcbo
        print(f'\nFinal Result (Iteration {bcbo_results[-1]["iteration"]}):')
        print(f'  BCBO:    {final_bcbo:.6f}')
        print(f'  BCBO-DE: {final_bcbode:.6f}')
        print(f'  Difference: {final_diff:.6f} ({"BCBO-DE better [WIN]" if final_diff > 0 else "BCBO better"})')
        print(f'  Improvement: {abs(final_diff/final_bcbo*100):.2f}%')

        if final_diff > 0:
            summary['bcbode_wins'] += 1
        else:
            summary['bcbo_wins'] += 1
        summary['total'] += 1

    elif 'M' in bcbo_results[0]:
        # Task scale comparison
        print(f'{"M (Tasks)":<12} {"BCBO Fitness":<18} {"BCBO-DE Fitness":<18} {"Difference":<15} {"Winner"}')
        print('-'*80)

        local_bcbo_wins = 0
        local_bcbode_wins = 0

        for b, bd in zip(bcbo_results, bcbode_results):
            diff = bd['best_fitness'] - b['best_fitness']
            winner = 'BCBO-DE [WIN]' if diff > 0 else 'BCBO'
            print(f'{b["M"]:<12} {b["best_fitness"]:<18.4f} {bd["best_fitness"]:<18.4f} {diff:<15.4f} {winner}')

            if diff > 0:
                local_bcbode_wins += 1
            else:
                local_bcbo_wins += 1

        print(f'\nSummary for this dataset:')
        print(f'  BCBO wins: {local_bcbo_wins}/{len(bcbo_results)}')
        print(f'  BCBO-DE wins: {local_bcbode_wins}/{len(bcbo_results)}')

        if local_bcbode_wins > local_bcbo_wins:
            summary['bcbode_wins'] += 1
        else:
            summary['bcbo_wins'] += 1
        summary['total'] += 1

print('\n' + '='*80)
print('OVERALL SUMMARY')
print('='*80)
print(f'Datasets where BCBO-DE performed better: {summary["bcbode_wins"]}/{summary["total"]}')
print(f'Datasets where BCBO performed better: {summary["bcbo_wins"]}/{summary["total"]}')
print(f'\nConclusion: {"BCBO-DE is SUPERIOR" if summary["bcbode_wins"] > summary["bcbo_wins"] else "BCBO is SUPERIOR" if summary["bcbo_wins"] > summary["bcbode_wins"] else "Mixed results"}')
