TARGET=image_filter
SRC=src/main.cu src/filters.cu
CXXFLAGS=`pkg-config --cflags opencv4`
LDFLAGS=`pkg-config --libs opencv4`

all:
	nvcc -o $(TARGET) $(SRC) $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
