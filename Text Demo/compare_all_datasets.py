# -*- coding: utf-8 -*-
"""
逐步分析四组数据中BCBO vs BCBO-DE的性能对比
"""

import json
import numpy as np

print('='*80)
print('Four Datasets BCBO vs BCBO-DE Performance Analysis')
print('='*80)
print()

all_results = []

for i in range(1, 5):
    chart_set = f'chart_set_{i}'
    json_file = f'RAW_data/{chart_set}_merged_results.json'
    
    print(f'{"="*80}')
    print(f'Dataset {i}: {chart_set}')
    print(f'{"="*80}')
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        config = data['config']
        print(f'Variable: {config["variable_param"]}')
        print(f'Data points: {len(config["values"])}')
        print(f'Runs per point: {config["runs_per_point"]}')
        print()
        
        algorithms = data['algorithms']
        
        if 'BCBO' not in algorithms or 'BCBO-DE' not in algorithms:
            print('[ERROR] Missing BCBO or BCBO-DE data')
            all_results.append({
                'chart_set': i,
                'status': 'ERROR',
                'message': 'Data missing'
            })
            continue
        
        bcbo_results = algorithms['BCBO']['results']
        bcbo_de_results = algorithms['BCBO-DE']['results']
        
        print('Performance Comparison:')
        print('-'*80)
        
        # Calculate all improvements
        all_improvements = []
        for idx in range(len(bcbo_results)):
            bcbo_val = bcbo_results[idx]['execution_time']
            bcbo_de_val = bcbo_de_results[idx]['execution_time']
            improvement = (bcbo_val - bcbo_de_val) / bcbo_val * 100
            all_improvements.append(improvement)
        
        avg_improvement = np.mean(all_improvements)
        std_improvement = np.std(all_improvements)
        min_improvement = np.min(all_improvements)
        max_improvement = np.max(all_improvements)
        
        print(f'Statistics (all {len(bcbo_results)} points):')
        print(f'  Avg improvement: {avg_improvement:+.2f}%')
        print(f'  Std dev: {std_improvement:.2f}%')
        print(f'  Min improvement: {min_improvement:+.2f}%')
        print(f'  Max improvement: {max_improvement:+.2f}%')
        print()
        
        bcbo_de_better = sum(1 for imp in all_improvements if imp > 0)
        bcbo_better = sum(1 for imp in all_improvements if imp < 0)
        
        print(f'Point statistics:')
        print(f'  BCBO-DE better: {bcbo_de_better}/{len(bcbo_results)} ({bcbo_de_better/len(bcbo_results)*100:.1f}%)')
        print(f'  BCBO better: {bcbo_better}/{len(bcbo_results)} ({bcbo_better/len(bcbo_results)*100:.1f}%)')
        print()
        
        if bcbo_de_better == len(bcbo_results):
            conclusion = '[PERFECT] BCBO-DE better at ALL points'
            status = 'PERFECT'
        elif bcbo_de_better >= len(bcbo_results) * 0.95:
            conclusion = f'[EXCELLENT] BCBO-DE better at {bcbo_de_better/len(bcbo_results)*100:.1f}% points'
            status = 'EXCELLENT'
        elif bcbo_de_better >= len(bcbo_results) * 0.8:
            conclusion = f'[GOOD] BCBO-DE better at {bcbo_de_better/len(bcbo_results)*100:.1f}% points'
            status = 'GOOD'
        elif avg_improvement > 0:
            conclusion = f'[MIXED] BCBO-DE better on average but unstable'
            status = 'MIXED'
        else:
            conclusion = f'[FAIL] BCBO-DE NOT better than BCBO'
            status = 'FAIL'
        
        print(f'Conclusion: {conclusion}')
        print()
        
        all_results.append({
            'chart_set': i,
            'status': status,
            'avg_improvement': avg_improvement,
            'std_improvement': std_improvement,
            'bcbo_de_better_rate': bcbo_de_better / len(bcbo_results) * 100,
            'conclusion': conclusion
        })
        
    except Exception as e:
        print(f'[ERROR] Analysis failed: {e}')
        all_results.append({
            'chart_set': i,
            'status': 'ERROR',
            'message': str(e)
        })
    
    print()

print('='*80)
print('SUMMARY REPORT: Overall Evaluation')
print('='*80)
print()

for result in all_results:
    i = result['chart_set']
    status = result['status']
    
    if status == 'ERROR':
        print(f'[Set {i}] ERROR: {result.get("message", "Unknown error")}')
    elif status == 'FAIL':
        print(f'[Set {i}] FAIL: {result["conclusion"]}')
    else:
        avg_imp = result['avg_improvement']
        better_rate = result['bcbo_de_better_rate']
        print(f'[Set {i}] {status}: Avg={avg_imp:+.2f}%, Better rate={better_rate:.1f}%')

print()
print('-'*80)
print('FINAL CONCLUSION:')
print('-'*80)

perfect = sum(1 for r in all_results if r['status'] == 'PERFECT')
excellent = sum(1 for r in all_results if r['status'] == 'EXCELLENT')
good = sum(1 for r in all_results if r['status'] == 'GOOD')
success = sum(1 for r in all_results if r['status'] not in ['ERROR', 'FAIL'])

if perfect == 4:
    print('[PERFECT] All 4 datasets: BCBO-DE better at ALL points')
    print('          -> Very stable and reliable algorithm')
elif excellent + perfect >= 3:
    print('[EXCELLENT] At least 3 datasets: BCBO-DE better at 95%+ points')
    print('            -> Excellent performance, suitable for publication')
elif success >= 3:
    print('[GOOD] At least 3 datasets: BCBO-DE better on average')
    print('       -> Clear improvement, stability needs attention')
else:
    print('[WARNING] Multiple datasets show poor BCBO-DE performance')
    print('          -> Check algorithm implementation or parameters')

print()
print('='*80)
