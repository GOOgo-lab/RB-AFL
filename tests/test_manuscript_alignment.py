from __future__ import annotations
import importlib.util
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
HAS_FULL = all(importlib.util.find_spec(m) for m in ("torch", "shapely", "geopandas", "PIL", "cryptography"))
if HAS_FULL:
    from shapely.geometry import Point, LineString, Polygon
    from rbafl.fields import fields_from_geometry
    from rbafl.watermark import nc_score, create_registry_record, recover_watermark, verify_recovered


@unittest.skipUnless(HAS_FULL, "Requires full GIS/model dependencies")
class EquationTests(unittest.TestCase):
    def test_point_distance_and_occupancy_use_grid_centers(self):
        t, m = fields_from_geometry([Point(0,0)], 3, 1, limit=1.5, distance_sigma_px=1)
        self.assertAlmostEqual(float(t[1,1,2]), float(np.exp(-0.5)), places=6)
        self.assertEqual(t[0,1,1],1)
        self.assertEqual(t[0,1,2],0)
        self.assertTrue(np.all(t[2]==0.5))

    def test_direction_is_directed_nearest_segment(self):
        # A right-to-left segment has theta=pi, not the historical axial angle 0.
        t,_=fields_from_geometry([LineString([(1,0),(-1,0)])],3,1,limit=1.5)
        np.testing.assert_allclose(t[2],1)

    def test_density_uses_each_segment_midpoint_not_centroid(self):
        geom=LineString([(-1,0),(0,0),(0,1)])
        t,_=fields_from_geometry([geom],3,1,limit=1.5)
        x,y=np.meshgrid([-1,0,1],[1,0,-1])
        raw=np.exp(-((x+0.5)**2+y*y)/2)+np.exp(-(x*x+(y-0.5)**2)/2)
        np.testing.assert_allclose(t[3],raw/(raw.max()+1e-12),rtol=1e-6)
        self.assertGreater(float(t[3].min()),0)  # no min subtraction

    def test_polygon_interior_is_distance_zero_and_holes_are_preserved(self):
        p=Polygon([(-2,-2),(2,-2),(2,2),(-2,2)],holes=[[(-.6,-.6),(.6,-.6),(.6,.6),(-.6,.6)]])
        t,_=fields_from_geometry([p],3,1,limit=1.5,distance_sigma_px=1)
        self.assertEqual(t[1,1,2],1)
        self.assertLess(t[1,1,1],1)
        self.assertEqual(t[0,1,1],0)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError): fields_from_geometry([],3)
        with self.assertRaises(ValueError): fields_from_geometry([Point(0,0)],3,0)


@unittest.skipUnless(HAS_FULL, "Requires full GIS/model dependencies")
class WatermarkTests(unittest.TestCase):
    def test_zero_reference_rejected_and_zero_recovery_is_zero(self):
        with self.assertRaises(ValueError): nc_score([0,0],[0,0])
        self.assertEqual(nc_score([1,0],[0,0]),0)

    def test_sparse_watermark_xor_round_trip_and_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'source.json';f.write_text('{}')
            model=Path(folder)/'weights';model.write_bytes(b'synthetic')
            w=np.zeros(256,dtype=np.uint8);w[:45]=1
            b=np.arange(256)%2
            record=create_registry_record('id',f,model,w,b,{'exp_id':'E5'},{'grid_size':256},'median')
            recovered=recover_watermark(record,b)
            np.testing.assert_array_equal(recovered,w)
            self.assertAlmostEqual(nc_score(w,recovered),1)
            self.assertFalse(verify_recovered(record,np.zeros(256),w,0)['passed'])
            changed=copy.deepcopy(record);changed['threshold_mode']='zero'
            with self.assertRaises(ValueError): recover_watermark(changed,b)
            changed=copy.deepcopy(record);changed['zero_watermark_bits_b64']='AA=='
            with self.assertRaises(ValueError): recover_watermark(changed,b)

    def test_median_ties_and_dimension(self):
        from rbafl.model import feature_to_bits
        x=np.array([0,1,1,2],dtype=np.float32)
        np.testing.assert_array_equal(feature_to_bits(x,4,'median'),[0,1,1,1])
        with self.assertRaises(ValueError): feature_to_bits(x,256)
        with self.assertRaises(ValueError): feature_to_bits([0,np.nan],2)


@unittest.skipUnless(HAS_FULL, "Requires full GIS/model dependencies")
class ProtocolRevisionTests(unittest.TestCase):
    def test_fixed_threshold_stays_fixed_even_if_far_target_fails(self):
        from rbafl.calibration import calibrate_threshold
        scores=pd.DataFrame({'score_type':['genuine','impostor'],'nc':[.7,.9]})
        with tempfile.TemporaryDirectory() as d:
            s=calibrate_threshold(scores,d,fixed_threshold=.75)
        self.assertEqual(s['selected_threshold'],.75)
        self.assertEqual(s['selection_policy'],'predeclared_fixed')
        self.assertEqual(s['false_accept_count'],1)
        self.assertEqual(s['false_reject_count'],1)

    def test_only_four_manuscript_attacks(self):
        from rbafl.protocol import attack_plan
        self.assertEqual({c.attack for c in attack_plan()},{'rotation','translation','scale','object_delete'})

    def test_calibrate_stage_calls_fixed_policy(self):
        from rbafl.study import load_config, calibrate_stage, config_paths
        c=load_config(Path(__file__).resolve().parents[1]/'configs/protocol_50_20_41.json')
        with tempfile.TemporaryDirectory() as folder:
            c['paths']['run_root']=folder
            output=config_paths(c)['calibration']
            scores=pd.DataFrame({'score_type':['genuine','impostor'],'nc':[.9,.2]})
            w=np.r_[np.ones(45,dtype=np.uint8),np.zeros(211,dtype=np.uint8)]
            with patch('rbafl.study.prepare_stage'),patch('rbafl.study.validate_split_file',return_value={'sha256':'synthetic'}),patch('rbafl.study.write_protocol_lock'),patch('rbafl.study._validate_eligible_watermark',return_value=w),patch('rbafl.study._e5_checkpoints',return_value={}),patch('rbafl.calibration.collect_calibration_scores',return_value=scores):
                result=calibrate_stage(c)
            payload=json.loads(result.read_text())
            self.assertEqual(payload['selected_threshold'],.75)
            self.assertTrue((output/'threshold_lock.sha256').is_file())


@unittest.skipUnless(HAS_FULL, "Requires full GIS/model dependencies")
class CenterTests(unittest.TestCase):
    def test_dual_signature_and_timestamp_tamper(self):
        from rbafl.signing import generate_keypair,sign_record,issue_center_record,verify_center_record
        from cryptography.exceptions import InvalidSignature
        with tempfile.TemporaryDirectory() as folder:
            r=Path(folder)
            generate_keypair(r/'user',r/'user.pub')
            generate_keypair(r/'center',r/'center.pub')
            signed=sign_record({'record_id':'synthetic'},r/'user',signer_id='registrant')
            envelope=issue_center_record(signed,r/'user.pub',r/'center',center_id='test-center',certificate_reference='test-only')
            self.assertTrue(verify_center_record(envelope,r/'center.pub',r/'user.pub'))
            envelope['timestamp_utc']='2000-01-01T00:00:00+00:00'
            with self.assertRaises(InvalidSignature): verify_center_record(envelope,r/'center.pub',r/'user.pub')


@unittest.skipUnless(HAS_FULL, "Requires full GIS/model dependencies")
class PipelineSmokeTests(unittest.TestCase):
    def test_train_register_verify_tiny_synthetic_data(self):
        import torch
        from PIL import Image
        from rbafl.model import train_ablation, TrainingConfig
        from rbafl.protocol import get_ablation
        from rbafl.evaluation import register_one, verify_one
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);prepared=root/'prepared';prepared.mkdir()
            rows=[]
            for identity in ['a','b']:
                for j,kind in enumerate(['base','aug_001','aug_002']):
                    tensor_path=prepared/f'{identity}_{kind}.npy'
                    np.save(tensor_path,np.random.default_rng(j+(10 if identity=='b' else 0)).random((4,16,16)).astype('float32'))
                    rows.append({'identity':identity,'sample_type':kind,'tensor_path':str(tensor_path),'source_path':f'{identity}.geojson'})
            pd.DataFrame(rows).to_csv(prepared/'manifest.csv',index=False)
            checkpoint=train_ablation(prepared,root/'models',get_ablation('E5'),TrainingConfig(epochs=1,batch_size=2,device='cpu'),reuse_existing=False)
            vector=root/'vector.geojson'
            vector.write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':{},'geometry':{'type':'LineString','coordinates':[[0,0],[1,.2],[.7,1]]}}]}))
            w=np.zeros(256,dtype='uint8');w[:45]=255;image=root/'watermark.png';Image.fromarray(w.reshape(16,16)).save(image)
            registry=root/'registry.json'
            register_one(vector,image,checkpoint,registry,identity='a',grid_size=16,timing_repeats=1,device='cpu')
            result=verify_one(vector,image,checkpoint,registry,'a',nc_threshold=.75,device='cpu')
            self.assertTrue(result['passed']);self.assertAlmostEqual(result['nc'],1)

@unittest.skipUnless(HAS_FULL, "Requires full GIS/model dependencies")
class CalibrationCohortTests(unittest.TestCase):
    def test_impostors_are_base_only_not_augmented(self):
        from rbafl.calibration import collect_calibration_scores
        ids=[f'id{i}' for i in range(20)]
        w=np.r_[np.ones(45,dtype=np.uint8),np.zeros(211,dtype=np.uint8)]
        rows=[{'identity':identity,'sample_type':kind,'tensor_path':'synthetic.npy'}
              for identity in ids for kind in ['base','aug_001']]
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'synthetic.pt';checkpoint.write_bytes(b'not-loaded')
            with patch('rbafl.calibration.identities_for_split',return_value=ids),patch('rbafl.calibration.subset_manifest_by_identities',return_value=pd.DataFrame(rows)),patch('rbafl.calibration.watermark_image_to_bits',return_value=(w,16,16)),patch('rbafl.calibration.load_encoder',return_value=(None,{'experiment':{'exp_id':'E5','channel_indices':[0,1,2,3]}},'cpu')),patch('rbafl.calibration.resolve_prepared_tensor_path',return_value=checkpoint),patch('rbafl.calibration.np.load',return_value=np.zeros((4,2,2))),patch('rbafl.calibration.extract_embedding',return_value=np.arange(256,dtype=np.float32)):
                result=collect_calibration_scores(folder,'unused',{seed:checkpoint for seed in range(20260730,20260740)},'unused',folder,random_watermark_patterns=0)
            self.assertEqual(sum(result.score_type=='impostor'),3800)
            self.assertEqual(set(result.loc[result.score_type=='impostor','sample_type']),{'base'})
            self.assertEqual(sum(result.score_type=='genuine'),200)

    def test_versioned_checkpoint_rejects_previous_fields(self):
        import torch
        from rbafl.model import load_encoder
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'old.pt';torch.save({'format':'rbafl_geometry_encoder_v1.0.0'},p)
            with self.assertRaises(ValueError):load_encoder(p,'cpu')

@unittest.skipUnless(HAS_FULL, "Requires full GIS/model dependencies")
class DiscussionExportTests(unittest.TestCase):
    def test_discussion_clean_self_cohort_is_separate(self):
        from rbafl.study import load_config, config_paths, uniqueness_stage
        c=load_config(Path(__file__).resolve().parents[1]/'configs/protocol_50_20_41.json')
        with tempfile.TemporaryDirectory() as folder:
            c['paths']['run_root']=folder; paths=config_paths(c)
            rows=[{'exp_id':'E5','registered_identity':str(i),'tested_identity':str(j),'nc':1.0 if i==j else .3}
                  for i in range(41) for j in range(41)]
            for seed in c['study']['training_seeds']:
                directory=paths['evaluation']/f'seed_{seed}';directory.mkdir(parents=True)
                pd.DataFrame(rows).to_csv(directory/'uniqueness_rows_all.csv',index=False)
            paths['calibration'].mkdir()
            pd.DataFrame({'score_type':['impostor']*3800,'nc':[.3]*3800}).to_csv(paths['calibration']/'calibration_scores.csv',index=False)
            with patch('rbafl.study.write_protocol_lock'),patch('rbafl.study.load_frozen_threshold',return_value=.75):
                uniqueness_stage(c)
            summary=json.loads((paths['uniqueness']/'discussion_diagnostic.json').read_text())
            self.assertEqual(summary['clean_self_count'],410)
            self.assertEqual(summary['calibration_impostor_count'],3800)
            self.assertEqual(summary['clean_self_frr'],0)
            self.assertEqual(len(pd.read_csv(paths['uniqueness']/'uniqueness_unordered_pairs.csv')),8200)
