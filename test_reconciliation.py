import contextlib,csv,io,json,tempfile,unittest
from pathlib import Path
from merge_passages import merge as original
from merge_indexed import merge as indexed,FIELDS


def fixture(folder,extra=False,shift=0):
    folder.mkdir();events=[dict(zip(FIELDS,[1,'ltr',10,.4,1,1800,600]))]
    if extra:events.append(dict(zip(FIELDS,[2,'ltr',10,.4,2,1800,1000])))
    with (folder/'events.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(events)
    with (folder/'tracks.jsonl').open('w') as f:
        for frame in range(0,20,2):
            people=[dict(id=1,anchor=[1600+frame*20,600],xyxy=[1500+frame*20,550,1700+frame*20,1100])]
            if extra:people.append(dict(id=2,anchor=[1600+frame*20,1000],xyxy=[1500+frame*20,950,1700+frame*20,1500]))
            f.write(json.dumps(dict(frame=frame+shift,people=people,umbrellas=[],matches=[]))+'\n')


class IndexedMergeTests(unittest.TestCase):
    def test_same_identity_and_distinct_companion_match_original_policy(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root/'primary');fixture(root/'secondary',extra=True)
            with contextlib.redirect_stdout(io.StringIO()):
                original(root/'primary',root/'secondary',root/'old');d=indexed(root/'primary',root/'secondary',root/'new')
            self.assertEqual(d['events'],2);self.assertEqual(d['suppressed'],1)
            for name in ['events.csv','tracks.jsonl']:self.assertEqual((root/'old'/name).read_bytes(),(root/'new'/name).read_bytes())
            self.assertEqual(list((root/'new').glob('merge_*.sqlite')),[])

    def test_misaligned_stream_is_rejected_without_publishing_partial_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root/'primary');fixture(root/'secondary',shift=2)
            with self.assertRaisesRegex(ValueError,'align'):indexed(root/'primary',root/'secondary',root/'new')
            self.assertFalse((root/'new/events.csv').exists());self.assertFalse((root/'new/tracks.jsonl').exists())
            self.assertEqual(list((root/'new').glob('*.tmp')),[])

    def test_events_without_observations_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root/'primary');fixture(root/'secondary')
            (root/'secondary/tracks.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'trajectory'):indexed(root/'primary',root/'secondary',root/'new')


if __name__=='__main__':unittest.main()
