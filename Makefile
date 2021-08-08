include local.mk

CXX = g++
LD = g++


CXXFLAGS += -std=c++11
CXXFLAGS += -I/usr/local/include 
CXXFLAGS += -Wall -g -Wshadow  -O3
CXXFLAGS += -I${POLYGLOT}
# CXXFLAGS +=  -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
LDFLAGS += ${POLYGLOT}/polyglot.a

t : t.o
	${LD} t.o ${LDFLAGS} -o ${@}
