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
		.\tests\EmnistMLPTrainerTests.cpp \
		.\tests\FileMLPSerializerTests.cpp \
		.\tests\EmnistDatasetReaderTests.cpp \
		.\tests\MatrixModelTests.cpp

build: matrix_lib
	$(CXX) -O3 m.cpp $(SOURCES) $(MATRIX_LIB).a

tests:
	$(CXX) $(TESTS) $(SOURCES) $(MATRIX_LIB).a -lgtest -o $(TEST_EXEC)

gcov_report:
	$(CXX) --coverage -fno-elide-constructors $(TESTS) $(SOURCES) $(MATRIX_LIB).a -lgtest -o gcov_$(TEST_EXEC)
	./gcov_$(TEST_EXEC)
	lcov -d . -t "gcov_$(TEST_EXEC)" -c -o coverage_test.info -b .
	lcov --remove coverage_test.info "/c/Coding/msys64/*" -o coverage_test.info
	genhtml -o report coverage_test.info
	rm -rf *.gcda *.gcno

matrix_lib:
	$(CXX) $(CFLAGS) -c $(MATRIX_LIB).cc -o $(MATRIX_LIB).o
	ar rcs $(MATRIX_LIB).a $(MATRIX_LIB).o
	ranlib $(MATRIX_LIB).a
	rm -f $(MATRIX_LIB).o
