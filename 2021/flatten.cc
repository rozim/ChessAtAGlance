#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <vector>

#include "leveldb/db.h"
#include "leveldb/status.h"
#include "snappy.h"

using namespace std;


int main(int argc, char * argv[]) {
  time_t t1 = time(0L);

  FILE *f = fopen(argv[2], "w");
  if (f == NULL) {
    perror("foo");
    exit(1);
  }
  size_t bs = 1024 * 1024 * 16;
  char *buf = (char *) malloc(bs);
  setbuffer(f, buf, bs);

  leveldb::DB* db;
  leveldb::Status status = leveldb::DB::Open(leveldb::Options(), argv[1], &db);
  assert(status.ok());

  leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
  long n = 0;
  long mod = 1;
  std::string z;
  size_t orig = 0;
  size_t compressed = 0;
  
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    n++;
    string value = it->value().ToString();
    if ((n % mod) == 0) {
      time_t dt = time(0L) - t1;
      printf("%ld %ld %ld : %ld -> %ld\n", n, dt, value.size(), orig, compressed);
      mod *= 2;
    }
    int zlen = snappy::Compress(value.c_str(), value.size(), &z);
    fwrite(&zlen, sizeof(zlen), 1, f); // careful: use int, not size_t
    fwrite(z.c_str(), zlen, 1, f);
    orig += value.size();
    compressed += zlen;
  }
  printf("\n");
  time_t dt = time(0L) - t1;
  printf("%ld %ld (s)\n", n, dt);
  printf("Orig:       %ld\n", orig);
  printf("Compressed: %ld\n", compressed);
  fflush(f);
  fclose(f);
}
