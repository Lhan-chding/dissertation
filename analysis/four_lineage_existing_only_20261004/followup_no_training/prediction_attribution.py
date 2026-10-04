#!/usr/bin/env python3
"""Exact algebra of saved Z3 minus Z2; no fit, solve, tune or model execution."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root = args.root.resolve()
    out = Path(__file__).resolve().parent
    pilot = root / 'ssvc_flow/docs/prospective_selection/pilot_four_20261004'
    inputs = {}

    def record(path):
        data = path.read_bytes()
        inputs[str(path)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        return data

    def read(path):
        return json.loads(record(path))

    def write_csv(name, rows):
        with (out / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    version = read(pilot / 'CODE_VERSION.json')['git_commit']
    source_checks = {}
    for name in ('features.py', 'kernels.py', 'semantics.py', 'selectors.py'):
        rel = 'ssvc_flow/src/prospective_selection/' + name
        data = record(root / rel)
        original = subprocess.check_output(['git', 'show', f'{version}:{rel}'], cwd=root)
        assert data == original, f'Experiment source mismatch: {name}'
        source_checks[name] = {'experiment_commit': version, 'byte_identical': True,
                               'sha256': hashlib.sha256(data).hexdigest()}
    sys.path.insert(0, str(root / 'ssvc_flow'))
    from src.prospective_selection.features import LEVELS, PreDecisionPacket
    from src.prospective_selection.kernels import FeatureKernel, _dot

    compact = read(root / 'ssvc_flow/docs/prospective_selection/first_four_20261004/compact_input.json')
    packets = {(entry['origin'], level): packet for entry in compact['prestates']
               for level, packet in entry['packets'].items()}
    csv_path = pilot / 'all_candidate_predictions.csv'
    saved = {(row['level'], row['origin'], row['recipe']): float(row['predicted_delta_R0'])
             for row in csv.DictReader(record(csv_path).decode().splitlines())}
    actions = [f'R{i}' for i in range(8)]
    components = ('current', 'history', 'repair', 'repair_history', 'nuisance')
    rows, fold_checks, ranking_checks = [], [], []
    errors = dict(saved_replay=0., component_sum=0., forward_identity=0., reverse_identity=0.,
                  relative_BestStatic_identity=0., cross_shared_component=0.)
    bit_equal = 0
    for path2 in sorted((pilot / 'models').glob('*Z2*.json')):
        path3 = path2.with_name(path2.name.replace(LEVELS[2], LEVELS[3]))
        model2, model3 = read(path2), read(path3)
        fold2, fold3 = read(pilot / 'folds' / path2.name), read(pilot / 'folds' / path3.name)
        train2, train3 = model2['training_packets'], model3['training_packets']
        order = [p['origin_id'] for p in train2]
        assert order == [p['origin_id'] for p in train3]
        assert fold2['held_origins'] == fold3['held_origins']
        assert model2['best_static'] == model3['best_static']
        assert model2['alpha'] == model3['alpha'] == 10.0
        kernel2, kernel3 = (FeatureKernel.from_dict(m['kernel']) for m in (model2, model3))
        for key in ('current', 'history', 'nuisance'):
            assert kernel2.scales[key] == kernel3.scales[key], f'Shared scale mismatch: {key}'
        assert kernel2.nuisance_scaler == kernel3.nuisance_scaler
        assert kernel2.panel == kernel3.panel
        assert kernel2.strata == kernel3.strata
        assert kernel2.weights() == {'current': 1., 'history': .25, 'nuisance': .25}
        assert kernel3.weights() == {'current': .5, 'history': .125, 'nuisance': .25,
                                    'repair': .5, 'repair_history': .125}
        for level, train in ((LEVELS[2], train2), (LEVELS[3], train3)):
            for packet in train:
                assert packet == packets[packet['origin_id'], level]
        typed2, typed3 = ([PreDecisionPacket.from_dict(p) for p in t] for t in (train2, train3))
        emb2, emb3 = ([k.embedding(p) for p in t] for k, t in ((kernel2, typed2), (kernel3, typed3)))
        for e2, e3 in zip(emb2, emb3):
            for c in ('current', 'history', 'nuisance'):
                assert e2[c] == e3[c], f'Shared embedding mismatch: {c}'
        beta2, beta3 = (np.asarray(m['beta']) for m in (model2, model3))
        mean2, mean3 = (np.asarray(m['mean']) for m in (model2, model3))
        assert np.array_equal(mean2, mean3)
        default = model2['best_static']
        di = actions.index(default)
        fold_checks.append(dict(lineage=fold2['held_lineage'], train_origin_order=order,
                                shared_means_identical=True, shared_embeddings_identical=True,
                                shared_scalers_identical=True, shared_scales_identical=True,
                                Z2_weights=kernel2.weights(), Z3_weights=kernel3.weights(),
                                Z2_scales=kernel2.scales, Z3_scales=kernel3.scales,
                                n_fit=len(order), saved_alpha=10., best_static=default))
        for origin in fold2['held_origins']:
            q2, q3 = (PreDecisionPacket.from_dict(packets[origin, level]) for level in LEVELS[2:])
            e2, e3 = kernel2.embedding(q2), kernel3.embedding(q3)
            for c in ('current', 'history', 'nuisance'):
                assert e2[c] == e3[c]
            k2, k3 = kernel2.gram([q2], typed2)[0], kernel3.gram([q3], typed3)[0]
            c2, c3 = {}, {}
            for k, e, es, dest in ((kernel2, e2, emb2, c2), (kernel3, e3, emb3, c3)):
                for c, w in k.weights().items():
                    dest[c] = np.array([w * _dot(e[c], t[c]) / k.scales[c] for t in es])
            for k, cs in ((k2, c2), (k3, c3)):
                errors['component_sum'] = max(errors['component_sum'], float(np.max(np.abs(k-sum(cs.values())))))
            for c, factor in (('current', .5), ('history', .5), ('nuisance', 1.)):
                errors['cross_shared_component'] = max(errors['cross_shared_component'],
                    float(np.max(np.abs(c3[c] - factor * c2[c]))))
            deltas = {c: c3.get(c, np.zeros(len(order))) - c2.get(c, np.zeros(len(order))) for c in components}
            pred2, pred3 = mean2 + k2 @ beta2, mean3 + k3 @ beta3
            rank2 = sorted(actions, key=lambda a: (-pred2[actions.index(a)], a))
            rank3 = sorted(actions, key=lambda a: (-pred3[actions.index(a)], a))
            ranking_checks.append(dict(origin=origin, Z2_ranking=rank2, Z3_ranking=rank3,
                full_ranking_unchanged=rank2==rank3, best_static=default,
                both_top_equal_best_static=rank2[0]==rank3[0]==default))
            delta = pred3 - pred2
            terms = {'mean_change': mean3-mean2,
                     **{f'forward_{c}': dk @ beta2 for c, dk in deltas.items()},
                     'forward_beta_change': k3 @ (beta3-beta2),
                     **{f'reverse_{c}': dk @ beta3 for c, dk in deltas.items()},
                     'reverse_beta_change': k2 @ (beta3-beta2)}
            for path in ('forward', 'reverse'):
                total = terms['mean_change'] + sum(terms[f'{path}_{c}'] for c in components) + terms[f'{path}_beta_change']
                errors[path+'_identity'] = max(errors[path+'_identity'], float(np.max(np.abs(delta-total))))
                errors['relative_BestStatic_identity'] = max(errors['relative_BestStatic_identity'],
                    float(np.max(np.abs((delta-delta[di])-(total-total[di])))))
            for i, action in enumerate(actions):
                for level, pred in ((LEVELS[2], pred2), (LEVELS[3], pred3)):
                    errors['saved_replay'] = max(errors['saved_replay'], abs(pred[i]-saved[level, origin, action]))
                    bit_equal += int(float(pred[i]).hex() == saved[level, origin, action].hex())
                row = dict(origin=origin, lineage=fold2['held_lineage'], stage='early' if origin.endswith('t32') else 'late',
                           action=action, best_static=default, n_fit=len(order), saved_alpha_Z2=model2['alpha'], saved_alpha_Z3=model3['alpha'],
                           Z2_prediction_R0=float(pred2[i]), Z3_prediction_R0=float(pred3[i]),
                           delta_prediction_R0=float(delta[i]),
                           Z2_prediction_BestStatic=float(pred2[i]-pred2[di]),
                           Z3_prediction_BestStatic=float(pred3[i]-pred3[di]),
                           delta_prediction_BestStatic=float(delta[i]-delta[di]))
                for name, values in terms.items():
                    row[name+'_R0'] = float(values[i])
                    row[name+'_BestStatic'] = float(values[i]-values[di])
                for base in ('R0', 'BestStatic'):
                    for path in ('forward', 'reverse'):
                        reward = sum(row[f'{path}_{c}_{base}'] for c in ('current', 'history'))
                        repair = sum(row[f'{path}_{c}_{base}'] for c in ('repair', 'repair_history'))
                        row[f'{path}_reward_reweight_{base}'] = reward
                        row[f'{path}_repair_added_{base}'] = repair
                        row[f'{path}_reward_repair_net_{base}'] = reward + repair
                        row[f'{path}_reward_repair_opposite_sign_{base}'] = reward * repair < 0
                rows.append(row)
    assert len(rows) == 64 and all(err < 1e-12 for err in errors.values()), errors
    write_csv('saved_Z3_Z2_attribution.csv', rows)
    summaries = []
    for origin in ['ALL'] + sorted({r['origin'] for r in rows}):
        group = [r for r in rows if origin == 'ALL' or r['origin'] == origin]
        for base in ('R0', 'BestStatic'):
            for path in ('forward', 'reverse'):
                reward_abs = sum(abs(r[f'{path}_reward_reweight_{base}']) for r in group)
                repair_abs = sum(abs(r[f'{path}_repair_added_{base}']) for r in group)
                net_abs = sum(abs(r[f'{path}_reward_repair_net_{base}']) for r in group)
                denominator = reward_abs + repair_abs
                active = [r for r in group if abs(r[f'{path}_reward_reweight_{base}']) + abs(r[f'{path}_repair_added_{base}']) > 0]
                summaries.append(dict(origin=origin, baseline=base, path=path, n_actions=len(group),
                    n_nonzero_actions=len(active), opposite_sign_actions=sum(r[f'{path}_reward_repair_opposite_sign_{base}'] for r in active),
                    sum_abs_reward_reweight=reward_abs, sum_abs_repair_added=repair_abs,
                    sum_abs_reward_repair_net=net_abs,
                    aggregate_cancellation_fraction=1-net_abs/denominator if denominator else '',
                    max_abs_mean_change=max(abs(r[f'mean_change_{base}']) for r in group),
                    max_abs_prediction_difference=max(abs(r[f'delta_prediction_{base}']) for r in group),
                    mean_abs_prediction_difference=float(np.mean([abs(r[f'delta_prediction_{base}']) for r in group])),
                    max_abs_beta_change=max(abs(r[f'{path}_beta_change_{base}']) for r in group)))
    write_csv('saved_Z3_Z2_attribution_summary.csv', summaries)
    checks = dict(prediction_attribution_rows=len(rows), saved_predictions_replayed=128,
                  saved_replay_bit_equal_count=bit_equal, fold_count=len(fold_checks), max_abs_errors=errors,
                  all_identities_under_1e_minus_12=True)
    scope = dict(actual_inputs=inputs, experiment_source_checks=source_checks,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 current_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                 command=f'{sys.executable} {Path(__file__).resolve()} --root {root}', numpy_version=np.__version__,
                 checks=checks, fold_checks=fold_checks, ranking_checks=ranking_checks,
                 execution_scope=dict(new_training_runs=0,new_model_calls=0,new_samples=0,selector_refits=0,
                     hyperparameter_searches=0,new_ablation_runs=0,gpu_jobs_submitted=0,statistical_resamples=0,
                     forward_backward_Jacobian_calls=0,final_T_reads=0),
                 interpretation=['Exact algebraic attribution of two existing predictions only.',
                    'Forward and reverse paths are nonunique decompositions; components are not causal effects.',
                    'Saved weights, scales and coefficients only; no hybrid model selected or evaluated.',
                    'Cancellation is measured after multiplication by saved coefficients, not information redundancy.'],
                 missing_fields=[])
    (out/'attribution_scope.json').write_text(json.dumps(scope, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'checks': checks, 'ALL_summaries': [s for s in summaries if s['origin']=='ALL']}, indent=2))


if __name__ == '__main__':
    main()
