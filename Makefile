CXX      := g++
CXXFLAGS := -pedantic-errors -Wall -Wextra -std=c++2a -fconcepts -Werror -O3 -pthread #-g
LDFLAGS  := #-lstdc++ -lm
BUILD    := ./build
OBJ_DIR  := $(BUILD)/objs
BIN_DIR  := $(BUILD)/bin
TARGET   := mnist
INCLUDE  := -Iinclude/
SRC      := $(wildcard examples/*.cpp)

OBJECTS := $(SRC:%.cpp=$(OBJ_DIR)/%.o)

all: build $(BIN_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

$(BIN_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -o $(BIN_DIR)/$(TARGET) $(OBJECTS)

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
