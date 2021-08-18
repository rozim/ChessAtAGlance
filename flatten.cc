#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <vector>

#include "leveldb/db.h"
#include "leveldb/status.h"

#include "protobuf/io/gzip_stream.h"
#include "protobuf/io/zero_copy_stream.h"
#include "protobuf/io/zero_copy_stream_impl.h"
#include "protobuf/io/coded_stream.h"

using namespace std;

using namespace google::protobuf::io;

int main(int argc, char * argv[]) {
  time_t t1 = time(0L);

  //explicit GzipOutputStream(ZeroCopyOutputStream* sub_stream);

  //ZeroCopyOutputStream example;
  //int outfd = open("outfile", O_WRONLY);
  //ZeroCopyOutputStream* output = new FileOutputStream(outfd);

  printf("tick: %d\n", __LINE__);
  int fd = open("myfile", O_CREAT | O_WRONLY, 0644);
  if (fd < 0) {
    perror("myfile");
    exit(1);
  }
  printf("tick: %d\n", __LINE__);  
  ZeroCopyOutputStream* raw_output = new FileOutputStream(fd);
  printf("tick: %d\n", __LINE__);  
  CodedOutputStream* coded_output = new CodedOutputStream(raw_output);
  assert(coded_output != NULL);
  printf("tick: %d\n", __LINE__);  
  int magic_number = 1234;
  char text[] = "Hello world!";
  printf("tick: %d\n", __LINE__);    
  coded_output->WriteLittleEndian32(magic_number);
  printf("tick: %d\n", __LINE__);    
  coded_output->WriteVarint32(strlen(text));
  printf("tick: %d\n", __LINE__);    
  coded_output->WriteRaw(text, strlen(text));
  printf("tick: %d\n", __LINE__);  
  delete coded_output;
  delete raw_output;
  printf("tick: %d\n", __LINE__);  
  close(fd);

  exit(0);

  leveldb::DB* db;
  leveldb::Status status = leveldb::DB::Open(leveldb::Options(), "mega-v2-1.leveldb", &db);
  assert(status.ok());

  leveldb::Iterator* it = db->NewIterator(leveldb::ReadOptions());
  long n = 0;
  long mod = 1;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    n++;
    string value = it->value().ToString();
    if ((n % mod) == 0) {
      time_t dt = time(0L) - t1;
      printf("%ld %ld %ld\n", n, dt, value.size());
      mod *= 2;
    }
  }
  printf("\n");
  time_t dt = time(0L) - t1;
  printf("%ld %ld (s)\n", n, dt);
}
