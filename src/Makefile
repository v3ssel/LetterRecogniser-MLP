CXX = g++
CFLAGS = -std=c++17 -Wall -Werror -Wextra
MATRIX_LIB = core/matrix/libmatrix/matrix
TEST_EXEC = tests.exe

SOURCES = .\core\MultilayerPerceptron.cpp \
		  .\core\serializer\FileMLPSerializer.cpp \
		  .\core\training\EmnistMLPTrainer.cpp \
		  .\core\training\EmnistDatasetReader.cpp \
		  .\core\training\EmnistData.cpp \
		  .\core\matrix\MatrixModel.cpp \
		  .\core\matrix\MatrixLayer.cpp

TESTS = .\tests\main_test.cpp \
		.\tests\FileMLPSerializerTests.cpp \
		.\tests\EmnistDatasetReaderTests.cpp \
		.\tests\MatrixModelTests.cpp

build: matrix_lib
	g++ -O3 m.cpp $(SOURCES) \
			      $(MATRIX_LIB).a

test:
	g++ $(TESTS) \
		$(SOURCES) \
		$(MATRIX_LIB).a \
		-lgtest -o $(TEST_EXEC)

gcov_report:
	lcov -t "./$(TEST_EXEC)" -o test.info --no-external -c -d . --ignore-errors mismatch
	genhtml -o report test.info

matrix_lib:
	$(CXX) $(CFLAGS) -c $(MATRIX_LIB).cc -o $(MATRIX_LIB).o
	ar rcs $(MATRIX_LIB).a $(MATRIX_LIB).o
	ranlib $(MATRIX_LIB).a
	rm -f $(MATRIX_LIB).o
