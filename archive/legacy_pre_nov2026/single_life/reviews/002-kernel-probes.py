"""Independent task-002 review probes. Run from single_life with uv."""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from single_life.artifacts import ScriptedMeanUpdater
from single_life.cli import main
from single_life.contracts import content_hash
from single_life.fixtures import load_spec, spec_from_dict
from single_life.runner import run_lifetime

ROOT=Path(__file__).resolve().parents[1]

def spec():
    raw=load_spec(ROOT/"configs/dev_continuing.json").to_dict()
    raw.update(horizon=8, interaction_budget=8, compute_budget=20, agent="random",
               artifact_period=2, artifact_window=4)
    raw["segments"]=[{**raw["segments"][0],"end_index":8}]
    return spec_from_dict(raw)

class MutatingUpdater:
    def __init__(self):
        self.inner=ScriptedMeanUpdater(1,4)
    def apply(self,view,store):
        if view.decision_index==3:
            means=store.versions[0].content["arm_means"]
            means[next(iter(means))]=0.123456
            raise RuntimeError("failure after mutating nested historical content")
        return self.inner.apply(view,store)

def artifact_mutation():
    result=run_lifetime(spec(),updater=MutatingUpdater())
    prior=json.loads(result.actor_inputs[1])["accessible_artifacts"][0]
    final=result.artifacts[0]
    return {"status":result.status,
            "committed_index":result.committed_index,
            "historical_content_preserved":prior["content"]==final["content"],
            "final_artifact_hash_valid":content_hash(final["content"])==final["content_hash"]}

def updater_identity(tmp):
    checkpoint=tmp/"updater-checkpoint.json"
    cfg=spec()
    initial=ScriptedMeanUpdater(3,1)
    run_lifetime(cfg,updater=initial,stop_after=4,checkpoint_path=checkpoint)
    try:
        resumed=run_lifetime(cfg,resume_path=checkpoint)
    except Exception as exc:
        return {"changed_updater_accepted":False,"error":str(exc)}
    uninterrupted=run_lifetime(cfg,updater=ScriptedMeanUpdater(3,1))
    return {"changed_updater_accepted":True,
            "same_semantic_stream":resumed.semantic_stream()==uninterrupted.semantic_stream(),
            "uninterrupted_deployments":[a["deployment_decision_index"] for a in uninterrupted.artifacts],
            "resumed_deployments":[a["deployment_decision_index"] for a in resumed.artifacts]}

def interrupted_write(tmp):
    config=tmp/"interrupt-config.json"
    config.write_text(json.dumps(spec().to_dict()))
    run_dir=tmp/"interrupted-run"
    original=Path.write_text
    record={"injected":False}
    def truncate_then_interrupt(path,data,*args,**kwargs):
        if path==run_dir/"public_log.jsonl" and '"action_index":2' in data:
            before=path.read_text()
            record["prefix_had_event_1"]='"action_index":1' in before
            with path.open("w",encoding="utf-8"):
                pass
            record["injected"]=True
            raise KeyboardInterrupt("interruption after truncate, before write")
        return original(path,data,*args,**kwargs)
    try:
        with patch.object(Path,"write_text",truncate_then_interrupt):
            main(["run","--config",str(config),"--run-dir",str(run_dir)])
    except KeyboardInterrupt:
        pass
    raw=(run_dir/"public_log.jsonl").read_bytes()
    manifest=json.loads((run_dir/"manifest.json").read_text())
    record.update(public_log_bytes_after_interrupt=len(raw),
                  recorded_committed_event_count=manifest["actual_usage"]["observed_events"],
                  manifest_hash_matches=hashlib.sha256(raw).hexdigest()==manifest["hashes"]["public_log_sha256"])
    return record

def output_alias(tmp):
    config=tmp/"alias-config.json"
    config.write_text(json.dumps(spec().to_dict()))
    run_dir=tmp/"alias-run"
    with contextlib.redirect_stdout(io.StringIO()):
        code=main(["run","--config",str(config),"--run-dir",str(run_dir),
                   "--summary",str(run_dir/"public_log.jsonl")])
    manifest=json.loads((run_dir/"manifest.json").read_text())
    data=(run_dir/"public_log.jsonl").read_bytes()
    return {"exit_code":code,
            "public_log_matches_manifest":hashlib.sha256(data).hexdigest()==manifest["hashes"]["public_log_sha256"],
            "public_log_replaced_with_summary":b'"note": "Offline infrastructure smoke.' in data}

def probes():
    with tempfile.TemporaryDirectory(prefix="single-life-review002-") as tmp:
        root=Path(tmp)
        return {"artifact_rollback":artifact_mutation(),
                "updater_resume_identity":updater_identity(root),
                "interrupted_journal_write":interrupted_write(root),
                "output_path_alias":output_alias(root)}

if __name__=="__main__":
    print(json.dumps(probes(),indent=2,sort_keys=True))
