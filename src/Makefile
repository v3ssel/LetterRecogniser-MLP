PROJECTDIR = $(CURDIR)
BUILDPATH = build
BINPATH = bin/Release
BINPATH_TESTING = bin/Testing
TEST_EXEC = MLPTests.exe
DISTPATH = letter-recogniser-archive
GCOVAPPLE =	
MACCOMPILER = 

ifeq ($(shell uname), Darwin)
	MACCOMPILER = -DCMAKE_CXX_COMPILER="/usr/local/bin/g++"
	GCOVAPPLE = --ignore-errors inconsistent,mismatch
	TEST_EXEC = MLPTests
endif

ifeq ($(shell uname), Linux)
	TEST_EXEC = MLPTests
endif


all: install tests gcov_report clean check dist


install: $(BUILDPATH)-install
	cd $(BUILDPATH)-install && \
	cmake -DINSTALLING=ON ../ -G "Unix Makefiles" && \
	cmake --build . --config=Release
	echo "BUILD SUCCESSFUL: .exe placed in src/$(BINPATH) folder"


uninstall: clean
	rm -rf $(dir $(BINPATH))


dist: | $(DISTPATH)
	cp -R core/* $(DISTPATH)/src
	cp -R controller/* $(DISTPATH)/src
	cp -R gui/* $(DISTPATH)/src
	cp CMakeLists.txt $(DISTPATH)
	tar -cf $(DISTPATH).tar $(DISTPATH)
	gzip -9f $(DISTPATH).tar
	rm -rf $(DISTPATH)


tests: $(BUILDPATH)-testing
	cd $(BUILDPATH)-testing && \
	cmake -DTESTING=ON $(MACCOMPILER) ../ -G "Unix Makefiles" && \
	cmake --build . --config=RelWithDebInfo
	echo "BUILD SUCCESSFUL: .exe placed in src/$(BINPATH_TESTING) folder"


gcov_report: $(BUILDPATH)-coverage
	cd $(BUILDPATH)-coverage && \
	cmake -DTESTING=ON -DCOVERAGE=ON $(MACCOMPILER) ../ -G "Unix Makefiles" && \
	cmake --build . --config=RelWithDebInfo
	./$(BINPATH_TESTING)/$(TEST_EXEC) && \
	lcov --test-name "$(TEST_EXEC)" --capture --no-external --base-directory $(PROJECTDIR) --directory $(BUILDPATH)-coverage/ $(GCOVAPPLE) -o coverage_test.info && \
	genhtml -o report coverage_test.info $(GCOVAPPLE)
	mv report $(BINPATH_TESTING)/report
	echo "GCOV SUCCESSFUL: report placed in $(BINPATH_TESTING)/report folder"


check:
	clang-format --verbose -n */*.cpp */*.h
	clang-format --verbose -n */*/*.cpp */*/*.h


clean:
	rm -rf $(BUILDPATH)-* $(DISTPATH).tar.gz


$(BUILDPATH)-%:
	if [ ! -d "$@" ]; then mkdir -p $@; fi


$(DISTPATH):
	mkdir -p $@
	mkdir -p $@/src

