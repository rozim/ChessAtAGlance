#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <vector>

#include "leveldb/db.h"
#include "leveldb/status.h"

using namespace std;

int main(int argc, char * argv[]) {
  time_t t1 = time(0L);

  std::vector<leveldb::DB*> dbs;
  leveldb::DB* db;
  leveldb::Status status = leveldb::DB::Open(leveldb::Options(), "mega-v2-1.leveldb", &db);
  assert(status.ok());

  leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
  long n = 0;
  long mod = 1;
  long size = 0;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    n++;
    string key = it->key().ToString();
    string value = it->value().ToString();
    size += key.size() + value.size();
    if ((n % mod) == 0) {
      time_t dt = time(0L) - t1;
      printf("%ld %ld : %ld %ld -> %ld\n", n, dt, key.size(), value.size(), size);
      mod *= 2;
    }
  }
  printf("\n");
  time_t dt = time(0L) - t1;
  printf("%ld %ld (s) %ld (bytes)\n", n, dt, size);
}
