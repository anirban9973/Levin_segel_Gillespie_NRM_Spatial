CXX      = g++
CXXFLAGS = -std=c++20 -O3
TARGET   = single_realization
SRC      = Levin_Segel_single_realization.cpp
HEADER   = nrm_heap_header.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(HEADER)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
