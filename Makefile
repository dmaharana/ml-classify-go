.PHONY: all linux windows clean

APP_NAME := ml-classify-go
BUILD_DIR := bin
MAIN_PACKAGE := ./cmd/$(APP_NAME)

LDFLAGS := -ldflags="-s -w -X main.version=$(shell git describe --tags --always --dirty)"
GO_BUILD_FLAGS := -trimpath

all: linux windows

linux:
	@echo "Building for Linux..."
	GOOS=linux GOARCH=amd64 go build $(GO_BUILD_FLAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(APP_NAME)-linux $(MAIN_PACKAGE)
	@echo "Linux build complete: $(BUILD_DIR)/$(APP_NAME)-linux"

windows:
	@echo "Building for Windows..."
	GOOS=windows GOARCH=amd64 go build $(GO_BUILD_FLAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(APP_NAME)-windows.exe $(MAIN_PACKAGE)
	@echo "Windows build complete: $(BUILD_DIR)/$(APP_NAME)-windows.exe"

clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)
	@echo "Clean complete."
