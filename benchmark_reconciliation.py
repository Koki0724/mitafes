"""Measure one reconciliation child process, including its peak RSS."""
import argparse,json,resource,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['original','indexed'],required=True)
    p.add_argument('--primary',type=Path,required=True);p.add_argument('--secondary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).parent;module='merge_indexed.py' if a.method=='indexed' else 'merge_passages.py'
    start=time.perf_counter()
    subprocess.run([sys.executable,str(root/module),'--primary',str(a.primary),'--secondary',str(a.secondary),'--output',str(a.output)],check=True,stdout=subprocess.DEVNULL)
    result=dict(method=a.method,wall_seconds=time.perf_counter()-start,peak_child_rss_mb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024))
    (a.output/'resources.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
