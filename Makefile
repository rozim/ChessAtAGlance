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
# CXXFLAGS +=  -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free

LDFLAGS += ${POLYGLOT}/polyglot.a
LDFLAGS += -L${OPEN_SPIEL_LIB} -lopen_spiel

t : t.o
	${LD} t.o ${LDFLAGS} -o ${@}
