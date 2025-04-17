CXX = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -pedantic -O3

TARGET = ad
SRC = ad.cc test_var.cc

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

