"""Offline audit of saved task-005 evidence; no provider calls or replayed lives."""
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads(path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
registration = read(ROOT / 'experiments/005-gemini-lifetime/registration.json')
digest = hashlib.sha256()
for path in sorted((ROOT / 'src/single_life').rglob('*.py')):
    data = path.read_bytes()
    digest.update(path.relative_to(ROOT / 'src/single_life').as_posix().encode())
    digest.update(b'\0')
    digest.update(len(data).to_bytes(8, 'big'))
    digest.update(data)
assert digest.hexdigest() == registration['source']['source_sha256']
roots = [ROOT / 'runs/005-gemini-smoke']
roots += sorted((ROOT / 'runs/005-gemini-pilot').glob('*/*'))
roots += sorted((ROOT / 'runs/005-gemini-references').glob('*/*'))
results = []
files = {}
totals = Counter()
spend = Decimal(0)
for run in roots:
    manifest = read(run / 'manifest.json')
    config = read(run / 'config.json')
    assert manifest['status'] == 'completed'
    assert manifest['code']['source_sha256'] == digest.hexdigest()
    assert sha(run / 'config.json') == manifest['config_sha256']
    for field, filename in [('public_log_sha256','public_log.jsonl'), ('private_log_sha256','private_log.jsonl'), ('artifacts_sha256','artifacts.json')]:
        assert sha(run / filename) == manifest['hashes'][field]
    events = [json.loads(s) for s in (run / 'public_log.jsonl').read_text().splitlines()]
    events = [e for e in events if e['record_type'] == 'observed_event']
    private = [json.loads(s) for s in (run / 'private_log.jsonl').read_text().splitlines()]
    assert len(events) == config['horizon'] == len(private)
    expected_reward = Decimal(0)
    for t, (e, p) in enumerate(zip(events, private), 1):
        assert e['action_index'] == t
        segment = next(s for s in config['segments'] if s['start_index'] <= t <= s['end_index'])
        probability = segment['probabilities'][e['chosen_action']]
        material = f"v1|{config['innovation_seed']}|{config['lifetime_id']}|{t}|{e['chosen_action']}"
        u = (int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'big') >> 11) / float(1 << 53)
        assert e['observed_outcome'] == int(u < probability)
        assert p['evaluator_diagnostics']['innovation_u'] == u
        assert p['evaluator_state']['chosen_probability'] == probability
        expected_reward += Decimal(str(probability))
    reward = sum(e['observed_outcome'] for e in events)
    assert reward == manifest['actual_usage']['realized_reward']
    records = [read(p) for p in sorted((run / 'responses').glob('*.json'))]
    cost = Decimal(0)
    previous_note = None
    for t, r in enumerate(records, 1):
        assert r['decision_index'] == t and r['attempt'] == 1 and r['parser_status'] == 'ok'
        assert r['model_version'] == 'gemini-3.5-flash'
        instruction, payload = r['prompt'].split('\n\npublic_payload:\n')
        assert instruction == registration['instruction']
        payload = json.loads(payload)
        expected = {'decision_index':t,'interactions_remaining':config['horizon']-t+1,'observation_window':12,
            'observations':[{'action':e['chosen_action'],'action_index':e['action_index'],'reward':e['observed_outcome']} for e in events[max(0,t-13):t-1]],
            'permitted_action_ids':config['action_ids'],'retained_strategy_note':previous_note if manifest['note_carry'] else None}
        assert payload == expected
        assert 'SENTINEL_PRIVATE' not in r['prompt']
        reply = json.loads(r['response_text'])
        assert reply['action'] == r['parsed_action'] == events[t-1]['chosen_action']
        assert reply['strategy_note'][:1000] == r['parsed_note']
        previous_note = r['parsed_note']
        u = r['usage']
        assert u['status'] == 'available'
        assert u['input_tokens'] + u['output_tokens'] == u['total_tokens']
        assert 0 <= u['thoughts_tokens'] <= u['output_tokens'] <= 2048
        assert 0 <= u['input_tokens'] <= 4096
        price = (Decimal(u['input_tokens'])*Decimal('1.5') + Decimal(u['output_tokens'])*9)/1000000
        assert price == Decimal(u['cost_usd'])
        cost += price
        totals.update({'attempts':1,'input_tokens':u['input_tokens'],'output_tokens_including_thinking':u['output_tokens'],'thoughts_tokens':u['thoughts_tokens']})
        totals['max_input_tokens'] = max(totals['max_input_tokens'],u['input_tokens'])
        totals['max_output_tokens'] = max(totals['max_output_tokens'],u['output_tokens'])
    artifacts = read(run / 'artifacts.json')
    if records:
        assert len(records) == len(events)
        assert len(artifacts) == (len(records)-1 if manifest['note_carry'] else 0)
        for t, a in enumerate(artifacts,1):
            assert a['creation_cutoff'] == t-1
            assert a['deployment_decision_index'] == t+1
            assert a['content']['strategy_note'] == records[t-1]['parsed_note']
    assert cost == Decimal(manifest['actual_usage']['known_spend_usd'])
    spend += cost
    actions = Counter(e['chosen_action'] for e in events)
    row = {'run':manifest['run_id'],'reward':reward,'expected_reward_of_chosen_actions':str(expected_reward),'cost_usd':str(cost),'attempts':len(records),'actions':dict(actions)}
    if records:
        row['note_text_changes_after_first'] = sum(a['parsed_note'] != b['parsed_note'] for a,b in zip(records,records[1:]))
    results.append(row)
    for path in sorted(run.rglob('*')):
        if path.is_file(): files[path.relative_to(ROOT).as_posix()] = sha(path)
for world, info in registration['worlds'].items():
    assert sha(ROOT / info['path']) == info['sha256']
    for condition in ['recent','persistent']:
        config = read(ROOT / f'runs/005-gemini-pilot/{world}/{condition}/config.json')
        base = dict(info['spec'])
        base['agent'] = config['agent']
        assert config == base
progress = read(ROOT / 'experiments/005-gemini-lifetime/progress.json')
assert progress['completed_total'] == totals['attempts'] == 196
assert progress['retries'] == progress['failures'] == progress['unknown_attempts'] == 0
assert Decimal(progress['known_spend_usd']) == spend
for path in sorted((ROOT / 'experiments/005-gemini-lifetime').iterdir()):
    if path.is_file(): files[path.relative_to(ROOT).as_posix()] = sha(path)
output = {'status':'passed','source_sha256':digest.hexdigest(),'offline_tests':42,'provider_calls_during_audit':0,'totals':dict(totals),'estimated_cost_usd':str(spend),'checks':['saved prompts exactly equal own last-12 observed prefix plus permitted prior note','all model actions match scored observations','note timing and empty initial memory','all rewards recomputed from indexed innovations and private probabilities','paired world configs differ only in agent identity','source and manifest data hashes match','per-attempt costs and progress total reconcile'],'runs':results,'file_sha256':files}
(ROOT / 'reviews/005-verification.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in output.items() if k != 'file_sha256'},indent=2))
