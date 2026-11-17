import json,pickle,unittest
from pathlib import Path
import numpy as np
from engine import StreamingCounter
from occlusion_counter import OcclusionCounter
from robust_tracker import RobustPoseAnchorTracker
from merge_passages import same_passage

PROFILE=json.loads((Path(__file__).parent/'reference/profiles/glass_1219_heads_including_bottom.json').read_text())

class AccuracyTests(unittest.TestCase):
    def test_near_camera_fast_motion_keeps_identity(self):
        tracker=RobustPoseAnchorTracker(25,PROFILE)
        ids=[]
        for frame,x in enumerate([1100,1280,1460,1640,1820]):
            k=np.zeros((17,3));k[:5]=[x,1500,.9]
            people=tracker.update([[x-100,1420,x+100,1620,.9,0]],[k],frame*2)
            ids.append(people[0]['id'])
        self.assertEqual(len(set(ids)),1)

    def test_late_visible_person_requires_observations_on_both_sides(self):
        c=OcclusionCounter(PROFILE,2880,25)
        for frame,x in [(0,1740),(2,1705),(4,1660),(6,1610)]:c.update(1,x,600,frame)
        self.assertEqual([e['direction'] for e in c.events],['rtl'])
        for frame,x in [(8,1570),(10,1520)]:c.update(1,x,600,frame)
        self.assertEqual(len(c.events),1)

    def test_late_birth_on_one_side_does_not_invent_crossing(self):
        c=OcclusionCounter(PROFILE,2880,25)
        for frame,x in [(0,1710),(2,1670),(4,1620),(6,1570)]:c.update(1,x,600,frame)
        self.assertEqual(c.events,[])

    def test_short_jitter_and_large_vertical_jump_do_not_count(self):
        for values in [[(1740,600),(1710,605),(1690,600),(1750,600)],[(1740,600),(1705,900),(1650,950),(1600,1000)]]:
            c=OcclusionCounter(PROFILE,2880,25)
            for i,(x,y) in enumerate(values):c.update(1,x,y,2*i)
            self.assertEqual(c.events,[])

    def test_occluded_counter_checkpoint_preserves_pending_evidence(self):
        c=OcclusionCounter(PROFILE,2880,25);c.update(1,1740,600,0);c.update(1,1705,600,2)
        c=pickle.loads(pickle.dumps(c));c.update(1,1660,600,4);c.update(1,1610,600,6)
        self.assertEqual(len(c.events),1)

    def test_passages_are_matched_by_space_and_direction(self):
        a=dict(direction='ltr',seconds=2,frame=50)
        p={f:dict(anchor=[1600+f*3,600],xyxy=[1500,550,1800,1100]) for f in range(40,60,2)}
        q={f:dict(anchor=[1600+f*3,850],xyxy=[1500,800,1800,1400]) for f in range(40,60,2)}
        self.assertTrue(same_passage(a,p,a,p))
        self.assertFalse(same_passage(a,p,a,q))
        self.assertFalse(same_passage(a,p,a|dict(direction='rtl'),p))

if __name__=='__main__':unittest.main()
