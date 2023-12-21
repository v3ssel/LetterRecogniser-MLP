CXX = g++
CFLAGS = -std=c++17 -Wall -Werror -Wextra
MATRIX_LIB = core/matrix/libmatrix/matrix

SOURCES = .\core\MultilayerPerceptron.cpp \
		  .\core\serializer\FileMLPSerializer.cpp \
		  .\core\training\EmnistMLPTrainer.cpp \
		  .\core\training\EmnistDatasetReader.cpp \
		  .\core\training\EmnistData.cpp \
		  .\core\matrix\MatrixModel.cpp \
		  .\core\matrix\MatrixLayer.cpp

TESTS = .\tests\FileMLPSerializerTests.cpp \
		.\tests\EmnistDatasetReaderTests.cpp

build: matrix_lib
	g++ -O3 m.cpp $(SOURCES) \
			      $(MATRIX_LIB).a

test:
	g++ .\tests\main_test.cpp $(TESTS) \
							  $(SOURCES) \
							  $(MATRIX_LIB).a \
							  -lgtest -o tests.exe

matrix_lib:
	$(CXX) $(CFLAGS) -c $(MATRIX_LIB).cc -o $(MATRIX_LIB).o
	ar rcs $(MATRIX_LIB).a $(MATRIX_LIB).o
	ranlib $(MATRIX_LIB).a
	rm -f $(MATRIX_LIB).o
