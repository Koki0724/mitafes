"""Regression cases for missing heads, duplicates and close, crossing tracks."""
import json,unittest
from pathlib import Path
import numpy as np
from vectorized_tracker import VectorizedPoseAnchorTracker,batch_anchors,batch_predictions
from reference.pose_tracker import PoseAnchorTracker,head_anchor

PROFILE=json.loads((Path(__file__).parent/'reference/profiles/glass_1219_heads_including_bottom.json').read_text())


class VectorizedTrackerTests(unittest.TestCase):
    def test_umbrella_assignments_and_ambiguity_rejections(self):
        from engine import associate_umbrellas as frozen
        from vectorized_association import associate_umbrellas as vectorized
        rng=np.random.default_rng(945)
        for count in [0,1,15,30,60]:
            for umbrella_count in [0,1,4,15]:
                for repeat in range(3):
                    people=[];umbrellas=[]
                    for i in range(count):
                        x,y=rng.uniform([800,200],[2200,1200]);people.append(dict(id=i+1,xyxy=[x,y,x+150,y+350]))
                    for i in range(umbrella_count):
                        x,y=rng.uniform([800,200],[2200,1000]);umbrellas.append(dict(id=i+101,xyxy=[x,y,x+300,y+140]))
                    self.assertEqual(frozen(umbrellas,people,PROFILE),vectorized(umbrellas,people,PROFILE))

    def test_anchor_fallbacks_and_even_odd_medians(self):
        rng=np.random.default_rng(912)
        boxes=np.tile([100.,200.,400.,900.,.9,0],(42,1))
        points=rng.uniform([100,200,0],[400,500,1],(42,17,3))
        for i,p in enumerate(points):
            p[:5,2]=0;p[:i%6,2]=.9
            p[5:7,2]=.9 if i%2 else 0
        anchors,scales,methods=batch_anchors(boxes,points)
        for i,(b,p) in enumerate(zip(boxes,points)):
            a,s,m=head_anchor(b,p)
            np.testing.assert_array_equal(anchors[i],a)
            self.assertAlmostEqual(scales[i],s,places=12)
            self.assertEqual(methods[i],m)

    def test_close_tracks_with_duplicates_occlusions_and_empty_frames(self):
        rng=np.random.default_rng(354)
        scalar=PoseAnchorTracker(25,PROFILE);vector=VectorizedPoseAnchorTracker(25,PROFILE)
        origins=rng.uniform([900,100],[2200,1400],(60,2))
        velocities=rng.uniform([-8,-1],[8,1],(60,2))
        for frame in range(0,160,2):
            detections=[];points=[]
            for i,xy in enumerate(origins+frame*velocities):
                if frame in [40,42,44] or rng.random()<.12:continue
                x,y=xy;confidence=float(rng.uniform(.2,.99))
                k=np.zeros((17,3));k[:5]=np.c_[xy+rng.normal(0,3,(5,2)),rng.uniform(0,1,5)]
                k[5:7]=[[x-20,y+60,.8],[x+20,y+60,.8]]
                if i%5==0:k[:5,2]=0
                if i%11==0:k[5:7,2]=0
                d=[x-40,y-30,x+40,y+350,confidence,0]
                detections.append(d);points.append(k)
                if i%9==0:detections.append(d[:4]+[confidence*.9,0]);points.append(k.copy())
            self.assertEqual(scalar.update(detections,points,frame),vector.update(detections,points,frame),f'frame={frame}')
            tracks=list(scalar.tracks.values())
            if tracks:
                predicted=batch_predictions(tracks,frame/25+.08)
                for p,t in zip(predicted,tracks):
                    np.testing.assert_allclose(p,scalar.predict(t,frame/25+.08)[0],rtol=1e-14,atol=1e-12)


if __name__=='__main__':unittest.main()
