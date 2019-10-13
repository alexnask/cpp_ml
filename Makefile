CXX      := g++
CXXFLAGS := -pedantic-errors -Wall -Wextra -std=c++2a -fconcepts -Werror -pthread -g -O3
LDFLAGS  := #-lstdc++ -lm
BUILD    := ./build
OBJ_DIR  := $(BUILD)/objs
BIN_DIR  := $(BUILD)/bin
# TARGET   := mnist
INCLUDE  := -Iinclude/

EXA_SRC  := $(notdir $(wildcard examples/*.cpp))
EXA_TARGETS := $(EXA_SRC:%.cpp=$(BIN_DIR)/%)
EXA_OBJECTS := $(EXA_SRC:%.cpp=$(OBJ_DIR)/%.o)

all: build $(EXA_TARGETS)

.SECONDARY: $(EXA_OBJECTS)

$(OBJ_DIR)/%.o: examples/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

$(BIN_DIR)/%: $(OBJ_DIR)/%.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -o $@ $<

.PHONY: all build clean debug release

build:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(BIN_DIR)/*
