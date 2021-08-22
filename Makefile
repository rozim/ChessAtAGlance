include local.mk

CXX = g++
LD = g++

# CXXFLAGS += -std=c++11
#CXXFLAGS += -std=c++1z
CXXFLAGS += -std=c++17
CXXFLAGS += -I/usr/local/include 
CXXFLAGS +=  -g -Wshadow  -O3 # -Wall
CXXFLAGS += -Wdefaulted-function-deleted

CXXFLAGS += -I${POLYGLOT}
CXXFLAGS += -I${OPEN_SPIEL}
CXXFLAGS += -I${ABSIEL_CPP}
CXXFLAGS += -I/Users/dave/miniforge3/lib/python3.9/site-packages/tensorflow/include
CXXFLAGS += -I/usr/local/include
CXXFLAGS += -I/opt/homebrew/include
#CXXFLAGS += -I${TENSORFLOW_DIR2}
#CXXFLAGS += -I/Users/dave/Projects/tensorflow/bazel-bin/third_party/eigen3/include
# CXXFLAGS +=  -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free

LDFLAGS += ${POLYGLOT}/polyglot.a
LDFLAGS += -L${OPEN_SPIEL_LIB} -lopen_spiel
#LDFLAGS += -L/usr/local/lib -lleveldb
LDFLAGS += -L/Users/dave/miniforge3/lib/python3.9/site-packages/tensorflow -ltensorflow_framework
LDFLAGS += -L/opt/homebrew/lib
LDFLAGS += -lleveldb
LDFLAGS += -lsnappy
LDFLAGS += -lgflags

all : t gen leveldb_read flatten

t : t.o
	${LD} t.o ${LDFLAGS} -o ${@}

gen : gen.o
	${LD} gen.o ${LDFLAGS} -o ${@}
	@echo OK

leveldb_read : leveldb_read.o
	${LD} leveldb_read.o ${LDFLAGS} -o ${@}

flatten : flatten.o
	${LD} flatten.o ${LDFLAGS} -o ${@}
