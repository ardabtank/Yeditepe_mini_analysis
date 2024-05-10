CC = g++
CXXFLAGS = -std=c++11 -Wall
CFLAGS := $(shell root-config --cflags)
LDFLAGS:=  $(shell root-config --libs) -lHist
UMUTFLAGS:= $(shell root-config --cflags --libs)
SRCS = normalize.cpp
OBJS = $(SRCS:.cpp=.o)


all: myapp

myapp : runner.o tree_reader.o 
	${CC}  obj/runner.o obj/tree_reader.o ${LDFLAGS} -o runnner.exe
	



runner.o : src/runner.cpp
	${CC} -c src/runner.cpp ${CFLAGS} -o obj/runner.o
tree_reader.o : src/tree_reader.C
	${CC} -c src/tree_reader.C ${CFLAGS} -o obj/tree_reader.o
%.o: %.cpp
	$(CC) $(CXXFLAGS) $(CFLAGS) -c -o $@ $<

