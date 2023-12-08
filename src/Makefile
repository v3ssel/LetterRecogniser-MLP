CXX = g++
CFLAGS = -std=c++17 -Wall -Werror -Wextra
MATRIX_LIB = core/libmatrix/matrix

matrix_lib:
	$(CXX) $(CFLAGS) -c $(MATRIX_LIB).cc -o $(MATRIX_LIB).o
	ar rcs $(MATRIX_LIB).a $(MATRIX_LIB).o
	ranlib $(MATRIX_LIB).a
	rm -f $(MATRIX_LIB).o