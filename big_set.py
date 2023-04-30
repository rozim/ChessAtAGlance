from farmhash import FarmHash64
import resource

N = 100000000

# Note: python hash() is fine for time

baseline = False

all = set()
for i in range(N):
  if baseline:
    all.add(hash(str(i)))
  else:
    all.add(FarmHash64(str(i)))

maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024 // 1024

print('RSS: ', maxrss, 'LEN: ', len(all), baseline)
