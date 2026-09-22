"""Same-task benchmark; emits actual measurements, input hash and platform metadata."""
import argparse,csv,hashlib,json,platform,statistics,time,tracemalloc
from pathlib import Path
from prefix_indexes import CharacterPrefixIndex, TernaryPrefixIndex, LinearPrefixIndex, SortedPrefixIndex,RadixPrefixIndex,DoubleArrayPrefixIndex

def main():
    p=argparse.ArgumentParser();p.add_argument('csv',type=Path);p.add_argument('--limit',type=int,default=10000);p.add_argument('--output',type=Path,default=Path('prefix-results.json'));a=p.parse_args()
    if a.limit<=0:p.error('--limit must be positive')
    keys=[]
    with a.csv.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if 'title' not in (reader.fieldnames or []):p.error('CSV requires a title column')
        for row in reader:
            if row['title']:keys.append(row['title'])
            if len(keys)>=a.limit:break
    if not keys:p.error('CSV contains no titles')
    queries=sorted(set(key[:min(5,len(key))] for key in keys))[:100]+['__missing_prefix__']
    results=[]
    for cls in (SortedPrefixIndex,RadixPrefixIndex,DoubleArrayPrefixIndex,CharacterPrefixIndex,TernaryPrefixIndex,LinearPrefixIndex):
        tracemalloc.start();start=time.perf_counter();index=cls(keys);elapsed=time.perf_counter()-start
        retained,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        timings=[]
        for q in queries:
            expected=[s for s in sorted(set(keys)) if s.startswith(q)][:10]
            assert index.search(q)==expected
            for _ in range(30):
                start=time.perf_counter_ns();index.search(q);timings.append(time.perf_counter_ns()-start)
        results.append(dict(engine=cls.__name__,build_seconds=elapsed,retained_python_bytes=retained,peak_python_bytes=peak,median_query_ns=statistics.median(timings)))
        del index
    digest=hashlib.sha256()
    with a.csv.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(source_sha256=digest.hexdigest(),titles=len(keys),unique_titles=len(set(keys)),query_count=len(queries),python=platform.python_version(),platform=platform.platform(),ranking='lexicographic',results=results),indent=2))
    print(a.output)
if __name__=='__main__':main()
