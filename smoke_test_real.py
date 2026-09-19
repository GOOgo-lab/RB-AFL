#!/usr/bin/env python3
"""Real-data end-to-end smoke test for RB-AFL manuscript revision 1.1.0.

Place this beside the RB-AFL-Reproducibility folder, or pass --repo explicitly.
This tests execution and invariants, NOT manuscript accuracy or robustness.
Requires 2-3 real vector inputs. Each run uses copies in a new output directory.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import uuid

# This test prioritizes portability and reproducibility over benchmark throughput.
# Avoid conflicting Intel OpenMP runtimes in some Windows Anaconda/PyTorch setups.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_error(fn, exception_type):
    try:
        fn()
    except exception_type:
        return
    raise AssertionError(f"Expected {exception_type.__name__}, but operation succeeded")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def find_repo(explicit):
    here = Path(__file__).resolve().parent
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        candidates = [here, here / "RB-AFL-Reproducibility", here.parent,
                      here.parent / "RB-AFL-Reproducibility", Path.cwd(),
                      Path.cwd() / "RB-AFL-Reproducibility"]
    for candidate in candidates:
        if (candidate / "rbafl" / "fields.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate v1.1.0 source. Pass --repo PATH_TO_RB-AFL-Reproducibility")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Two or three real vector datasets")
    parser.add_argument("--watermark", required=True, help="Real experimental watermark image")
    parser.add_argument("--repo", help="Directory containing rbafl/ and pyproject.toml")
    parser.add_argument("--out", default="smoke_runs", help="Parent for new, uniquely named run directories")
    parser.add_argument("--grid-size", type=int, choices=(32, 64, 128, 256), default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")

    if len(args.inputs) not in (2, 3):
        parser.error("Provide exactly two or three datasets")
    run = Path(args.out).expanduser().resolve() / (
        "smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "rbafl_real_smoke_v1", "status": "RUNNING", "synthetic_only": False,
        "numerical_reproduction_verified": False,
        "parameters": vars(args), "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "mkl_threading_layer": os.environ.get("MKL_THREADING_LAYER")},
        "steps": [], "attacks": [],
        "limitations": ["Two or three real identities; no 50/20/41 study split or generalization test.",
                        "One E5 model; no ten-seed ablation, cache scheduler or timing benchmark.",
                        "Attacked-sample authentication may pass or fail; only execution and valid metrics are required.",
                        "The newly trained smoke-test model must not replace the paper's original model."],
    }
    start = time.perf_counter()

    def stage(name, fn):
        print(f"[RUN ] {name}", flush=True)
        begin = time.perf_counter()
        row = {"name": name, "status": "RUNNING"}
        report["steps"].append(row)
        write_json(run / "smoke_report.json", report)
        try:
            details = fn()
        except Exception as exc:
            row.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            row.update(status="PASS", details=details)
            print(f"[PASS] {name}", flush=True)
        finally:
            row["seconds"] = round(time.perf_counter() - begin, 6)
            write_json(run / "smoke_report.json", report)

    try:
        def preflight():
            repo = find_repo(args.repo)
            sys.path.insert(0, str(repo))
            # Import torch first; no package installation or network access occurs.
            versions = {}
            for name in ("torch", "numpy", "pandas", "scipy", "shapely", "geopandas", "PIL", "cryptography"):
                try:
                    module = importlib.import_module(name)
                except ImportError as exc:
                    raise RuntimeError(f"Missing/unusable dependency {name}. In the source directory run: python -m pip install -e \".[full,test]\"") from exc
                versions[name] = getattr(module, "__version__", "unknown")
            import rbafl
            require(rbafl.__version__ == "1.1.0", "This smoke test targets RB-AFL 1.1.0")
            require(Path(rbafl.__file__).resolve().parent == repo / "rbafl", "Imported the wrong rbafl package")
            report["source_sha256"] = {p.name: sha256(p) for p in sorted((repo / "rbafl").glob("*.py"))}
            report["environment"].update(versions)
            report["source_root"] = str(repo)
            return versions
        stage("01 dependencies and source version", preflight)

        import geopandas as gpd
        import numpy as np
        import torch
        from rbafl.data import prepare_dataset, PreparedTripletDataset, resolve_prepared_tensor_path
        from rbafl.model import TrainingConfig, train_ablation, load_encoder, extract_embedding, feature_to_bits
        from rbafl.protocol import get_ablation
        from rbafl.evaluation import register_one, verify_one
        from rbafl.vector import read_vector, apply_attack
        from rbafl.watermark import watermark_image_to_bits, validate_record_integrity, nc_score, recover_watermark
        from rbafl.signing import generate_keypair, sign_record, verify_record_signature, issue_center_record, verify_center_record
        from cryptography.exceptions import InvalidSignature

        torch.set_num_threads(args.threads)
        torch.manual_seed(20260730)
        raw = run / "raw_real_copies"
        raw.mkdir()
        registry = run / "registry.json"
        watermark = run / ("watermark" + Path(args.watermark).suffix)
        state = {}

        def copy_inputs():
            originals = [Path(p).expanduser().resolve() for p in args.inputs]
            require(len(set(originals)) == len(originals), "Inputs must be different datasets")
            state['input_hashes'] = {}
            state['sources'] = []
            metadata = []
            for i, original in enumerate(originals, 1):
                require(original.is_file(), f"Missing dataset: {original}")
                require(original.suffix.lower() in ('.shp','.geojson','.gpkg'), "Supported inputs: shp, geojson, gpkg")
                parts = [original]
                if original.suffix.lower() == '.shp':
                    parts = [p for p in original.parent.glob(original.stem + '.*') if p.is_file()]
                    require({'.shp','.shx','.dbf'} <= {p.suffix.lower() for p in parts}, "Incomplete Shapefile")
                for part in parts:
                    state['input_hashes'][str(part)] = sha256(part)
                    shutil.copy2(part, raw / (f'dataset_{i:02d}' + part.name[len(original.stem):]))
                target = raw / (f'dataset_{i:02d}' + original.suffix)
                original_gdf = gpd.read_file(target)
                gdf = read_vector(target)
                require(len(gdf) >= 2, "Dataset needs at least two usable features")
                require(gdf.crs is not None, "Dataset needs an explicit CRS")
                state['sources'].append(target)
                metadata.append({'identity':target.stem,'original_path':str(original),
                                 'readable_feature_count':len(original_gdf),'feature_count':len(gdf),
                                 'geometry_cleanup_removed_count':len(original_gdf)-len(gdf),'geometry_types':gdf.geom_type.value_counts().to_dict(),
                                 'crs':str(gdf.crs),'subset_or_simplification':False})
            original_watermark = Path(args.watermark).expanduser().resolve()
            state['input_hashes'][str(original_watermark)] = sha256(original_watermark)
            shutil.copy2(original_watermark, watermark)
            actual, _, _ = watermark_image_to_bits(watermark,256)
            require(0 < int(actual.sum()) < 256, "Watermark must contain both zero and one bits")
            state['watermark_hash'] = sha256(watermark)
            state['watermark_bits'] = actual
            state['identity_count'] = len(originals)
            write_json(run/'input_manifest.json', {'datasets':metadata,'sha256':state['input_hashes']})
            return {'datasets':metadata,'watermark_ones':int(actual.sum()),'watermark_zeros':256-int(actual.sum())}
        stage("02 copy and inspect real experimental inputs", copy_inputs)

        def prepare():
            prepared = run / "prepared"
            manifest = prepare_dataset(raw,prepared,grid_size=args.grid_size,
                                       density_sigma=3.0,augmentations_per_identity=2,seed=20260729)
            require(len(manifest)==3*state["identity_count"] and manifest.identity.nunique()==state["identity_count"], "Expected each identity x (base+2 augmentations)")
            for row in manifest.to_dict("records"):
                path = resolve_prepared_tensor_path(prepared,row["identity"],row["sample_type"],row["tensor_path"])
                tensor=np.load(path,allow_pickle=False)
                require(tensor.shape==(4,args.grid_size,args.grid_size), "Unexpected channel tensor shape")
                require(bool(np.isfinite(tensor).all()), "Nonfinite field value")
                require(float(tensor.min())>=0 and float(tensor.max())<=1, "Channel values must be in [0,1]")
                require(bool(np.isin(tensor[0],[0,1]).all()), "Occupancy is not binary")
            train=PreparedTripletDataset(prepared,('occ','dist','orient','density'),'train',validation_per_identity=1)
            val=PreparedTripletDataset(prepared,('occ','dist','orient','density'),'val',validation_per_identity=1)
            for cid in train.all_by_class:
                require(set(train.all_by_class[cid]).isdisjoint(val.all_by_class[cid]),"Internal holdout leakage")
            state.update(prepared=prepared,manifest=manifest)
            return {"tensor_count":len(manifest),"shape":[4,args.grid_size,args.grid_size],"train_samples":len(train),"internal_val_samples":len(val)}
        stage("03 real geometric fields and triplet preparation",prepare)

        def train():
            checkpoint=train_ablation(state['prepared'],run/'models',get_ablation('E5'),
                TrainingConfig(epochs=args.epochs,batch_size=3,embedding_dim=256,device=args.device,
                               seed=20260730,num_workers=0),reuse_existing=False)
            history=json.loads((checkpoint.parent/'history.json').read_text(encoding='utf-8'))
            require(len(history)==args.epochs,"Unexpected epoch count")
            for epoch in history:
                for split in ['train','val']:
                    require(all(math.isfinite(float(v)) for v in epoch[split].values()),"Nonfinite training metric")
            state['checkpoint']=checkpoint
            return {'checkpoint':str(checkpoint.relative_to(run)),'epochs':args.epochs,
                    'final_train_loss':history[-1]['train']['loss'],'final_val_loss':history[-1]['val']['loss']}
        stage("04 train and save a small E5 encoder",train)

        def embedding_check():
            encoder,ckpt,device=load_encoder(state['checkpoint'],args.device)
            require(ckpt['format']=='rbafl_geometry_encoder_v1.1.0',"Unexpected checkpoint schema")
            row=state['manifest'].iloc[0]
            path=resolve_prepared_tensor_path(state['prepared'],row['identity'],row['sample_type'],row['tensor_path'])
            tensor=np.load(path,allow_pickle=False)
            feature=extract_embedding(encoder,tensor,(0,1,2,3),device)
            require(feature.shape==(256,),"Wrong embedding dimension")
            require(bool(np.isfinite(feature).all()),"Nonfinite embedding")
            require(abs(float(np.linalg.norm(feature))-1)<1e-5,"Embedding not L2-normalized")
            bits=feature_to_bits(feature,256,'median')
            require(bool(np.array_equal(bits,(feature>=np.median(feature)).astype(np.uint8))),"Not median quantization")
            state['feature_bits']=bits
            return {'embedding_dim':256,'l2_norm':float(np.linalg.norm(feature)),'feature_one_bits':int(bits.sum()),'device':device}
        stage("05 load checkpoint and verify feature quantization",embedding_check)

        def clean_roundtrip():
            records=[]; results=[]
            for source in state['sources']:
                record=register_one(source,watermark,state['checkpoint'],registry,identity=source.stem,
                    grid_size=args.grid_size,threshold_mode='median',timing_repeats=1,device=args.device)
                result=verify_one(source,watermark,state['checkpoint'],registry,source.stem,
                    output_recovered_image=run/f'recovered_clean_{source.stem}.png',nc_threshold=.75,device=args.device)
                require(result['passed'] and abs(result['nc']-1)<1e-6 and result['ber']==0,"Clean self recovery failed")
                records.append(record);results.append(result)
            state['records']=records
            write_json(run/'clean_results.json',results)
            return [{'identity':r['identity'],'nc':r['nc'],'ber':r['ber']} for r in results]
        stage("06 register and verify all clean identities",clean_roundtrip)

        def attacked_roundtrip():
            directory=run/'attacks';directory.mkdir()
            for source in state['sources']:
                gdf=read_vector(source)
                for index,(attack,strength) in enumerate([('rotation',30.),('scale',1.3),('translation',.2),('object_delete',.3)]):
                    attacked,meta=apply_attack(gdf,attack,strength,seed=20260910+index)
                    path=directory/f'{source.stem}_{attack}.gpkg'
                    attacked.to_file(path, driver='GPKG', index=False)
                    result=verify_one(path,watermark,state['checkpoint'],registry,source.stem,
                        nc_threshold=.75,device=args.device)
                    require(math.isfinite(result['nc']) and 0<=result['nc']<=1,"Invalid attack NC")
                    require(math.isfinite(result['ber']) and 0<=result['ber']<=1,"Invalid attack BER")
                    report['attacks'].append({'identity':source.stem,'attack':attack,'strength':strength,
                                             'nc':result['nc'],'ber':result['ber'],'authentication_passed':result['passed'],
                                             'result_feature_count':meta['result_feature_count']})
            require(len(report['attacks']) == 4*state['identity_count'], 'Unexpected attack case count')
            write_json(run/'attack_results.json',report['attacks'])
            return {'completed':len(report['attacks']), 'expected':4*state['identity_count'],
                    'note':'Authentication is observational; no robustness target is asserted for a tiny model.'}
        stage("07 four attack families and watermark recovery",attacked_roundtrip)

        def integrity():
            record=state['records'][0]
            validate_record_integrity(record)
            changed=copy.deepcopy(record);changed['threshold_mode']='zero'
            expect_error(lambda:validate_record_integrity(changed),ValueError)
            changed=copy.deepcopy(record);changed['zero_watermark_bits_b64']='AA=='
            expect_error(lambda:recover_watermark(changed,state['feature_bits']),ValueError)
            require(nc_score(state['watermark_bits'],np.zeros(256,dtype=np.uint8))==0,"Zero recovery should score zero")
            expect_error(lambda:nc_score(np.zeros(256),np.zeros(256)),ValueError)
            return {'configuration_tamper_rejected':True,'zero_watermark_tamper_rejected':True,'zero_norm_handled':True}
        stage("08 integrity and zero-watermark edge cases",integrity)

        def signatures():
            # The signing API accepts filesystem paths; keep only generated PUBLIC
            # keys after the stage, and delete only the two files created here.
            keydir=run/'test_keys';keydir.mkdir()
            private_files=[keydir/'user_test.key',keydir/'center_test.key']
            try:
                generate_keypair(private_files[0],keydir/'user_test.pub.pem')
                generate_keypair(private_files[1],keydir/'center_test.pub.pem')
                signed=sign_record(state['records'][0],private_files[0],signer_id='smoke-user')
                require(verify_record_signature(signed,keydir/'user_test.pub.pem'),"User signature failed")
                envelope=issue_center_record(signed,keydir/'user_test.pub.pem',private_files[1],
                    center_id='smoke-center',certificate_reference='smoke-test-only')
                require(verify_center_record(envelope,keydir/'center_test.pub.pem',keydir/'user_test.pub.pem'),"Center signature failed")
                write_json(run/'signed_record.json',signed);write_json(run/'center_record.json',envelope)
                bad=copy.deepcopy(signed);bad['identity']='modified'
                expect_error(lambda:verify_record_signature(bad,keydir/'user_test.pub.pem'),InvalidSignature)
                bad=copy.deepcopy(envelope);bad['timestamp_utc']='2000-01-01T00:00:00+00:00'
                expect_error(lambda:verify_center_record(bad,keydir/'center_test.pub.pem',keydir/'user_test.pub.pem'),InvalidSignature)
            finally:
                for path in private_files:
                    if path.is_file():path.unlink()
            return {'user_signature':True,'center_signature':True,'tampering_rejected':True,'private_keys_retained':False}
        stage("09 user and center signature round trips",signatures)

        def originals_unchanged():
            for path,digest in state['input_hashes'].items():
                require(sha256(path)==digest,"Original source was modified")
            require(sha256(watermark)==state['watermark_hash'],"Watermark input was modified")
            return {'source_count':len(state['input_hashes']),'source_hashes_unchanged':True}
        stage("10 original inputs remain unchanged",originals_unchanged)
        report['status']='PASS'
    except Exception as exc:
        report['status']='FAIL'
        report['error']=f'{type(exc).__name__}: {exc}'
        report['traceback']=traceback.format_exc()
        print(f"[FAIL] {report['error']}",file=sys.stderr,flush=True)
    finally:
        report['elapsed_seconds']=round(time.perf_counter()-start,6)
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_json(run/'smoke_report.json',report)
        lines=['RB-AFL real-data smoke test',f"Status: {report['status']}",
               f"Elapsed seconds: {report['elapsed_seconds']}",'']
        lines.extend(f"{row['status']}: {row['name']} ({row.get('seconds',0):.3f}s)" for row in report['steps'])
        if report.get('error'):lines+=['',report['error'],report.get('traceback','')]
        lines+=['','Not a reproduction of manuscript results.']
        (run/'smoke_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(f"\n{report['status']} | report: {run / 'smoke_report.json'}",flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
