BUILDPATH = build
BINPATH = bin/Release
BINPATH_TESTING = bin/Testing
TEST_EXEC = MLPTests.exe
DISTPATH=letter-recogniser-archive
LINTER=../materials/linters/.clang-format


all: install clean


install: clean | $(BUILDPATH)
	cd $(BUILDPATH) && \
	cmake -DCMAKE_BUILD_TYPE=Release -DINSTALLING=ON ../ -G "Unix Makefiles" && \
	make -j4
	echo "BUILD SUCCESSFUL: .exe placed in src/$(BINPATH) folder"


uninstall: clean
	rm -rf $(dir $(BINPATH))


dvi:
	open ReadMe.html


dist: | $(DISTPATH)
	cp -R core/* $(DISTPATH)/src
	cp -R controller/* $(DISTPATH)/src
	cp -R LetterRecogniser/* $(DISTPATH)/src
	cp CMakeLists.txt $(DISTPATH)
	tar -cf $(DISTPATH).tar $(DISTPATH)
	gzip -9f $(DISTPATH).tar
	rm -rf $(DISTPATH)


tests: clean | $(BUILDPATH)
	cd $(BUILDPATH) && \
	cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTESTING=ON ../ -G "Unix Makefiles" && \
	make -j4
	echo "BUILD SUCCESSFUL: .exe placed in src/$(BINPATH_TESTING) folder"


gcov_report: clean | $(BUILDPATH)
	cd $(BUILDPATH) && \
	cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTESTING=ON -DCOVERAGE=ON ../ -G "Unix Makefiles" && \
	make -j4
	cd $(BINPATH_TESTING) && ./$(TEST_EXEC) && \
	lcov --test-name "$(TEST_EXEC)" --capture --directory ../../build/  -o coverage_test.info && \
	lcov --remove coverage_test.info "/c/Coding/msys64/*" -o coverage_test.info && \
	genhtml -o report coverage_test.info
	echo "GCOV SUCCESSFUL: report placed in src/$(BINPATH_TESTING)/report folder"
# --no-external - add this to first lcov
# and delete last


research: clean | $(BUILDPATH)
	cd $(BUILDPATH) && \
	cmake -DCMAKE_BUILD_TYPE=Release -DRESEARCH=ON ../ -G "Unix Makefiles" && \
	make -j4
	echo "BUILD SUCCESSFUL: .exe placed in src/$(BINPATH) folder"


check:
	cp $(LINTER) .
	clang-format -n */*.cpp */*.h
	clang-format -n */*/*.cpp */*/*.h
	clang-format -n */*/*/*.cpp */*/*/*.h
	rm -rf $(notdir $(LINTER))


clean:
	rm -rf $(BUILDPATH) $(DISTPATH).tar.gz


$(BUILDPATH):
	mkdir -p $@


$(DISTPATH):
	mkdir -p $@
	mkdir -p $@/src

