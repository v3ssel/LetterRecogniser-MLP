BUILDPATH = build
BINPATH = bin/Release
BINPATH_TESTING = bin/Testing
TEST_EXEC = MLPTests.exe

install: | $(BUILDPATH)
	cd $(BUILDPATH) && \
	cmake -DCMAKE_BUILD_TYPE=Release -DINSTALLING=ON ../ -G "Unix Makefiles" && \
	make -j4
	echo "BUILD SUCCESSFUL: .exe placed in src/$(BINPATH) folder"


tests: | $(BUILDPATH)
	cd $(BUILDPATH) && \
	cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTESTING=ON ../ -G "Unix Makefiles" && \
	make -j4
	echo "BUILD SUCCESSFUL: .exe placed in src/$(BINPATH_TESTING) folder"


gcov_report: | $(BUILDPATH)
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

clean:
	rm -rf $(BUILDPATH)


$(BUILDPATH):
	mkdir -p $@
