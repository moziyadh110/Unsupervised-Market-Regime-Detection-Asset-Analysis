.PHONY: install clean test help notebook lint format docker-build docker-up docker-down docker-test

help:
	@echo "Geopolitical Risk Analysis - Available Commands"
	@echo "================================================"
	@echo "make install       - Install project dependencies"
	@echo "make clean         - Remove cache and build files"
	@echo "make test          - Run unit tests"
	@echo "make notebook      - Start Jupyter notebook server"
	@echo "make lint          - Check code style (requires flake8)"
	@echo "make format        - Auto-format code (requires black)"
	@echo ""
	@echo "Docker Commands:"
	@echo "make docker-build  - Build Docker image"
	@echo "make docker-up     - Start containers (Jupyter on :8888)"
	@echo "make docker-down   - Stop containers"
	@echo "make docker-test   - Run tests in Docker"
	@echo "make docker-shell  - Open shell in container"
	@echo ""
	@echo "make help          - Show this help message"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete"

clean:
	@echo "Cleaning up cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/
	@echo "✓ Cleanup complete"

test:
	@echo "Running tests..."
	python -m pytest tests/ -v
	@echo "✓ Tests complete"

notebook:
	@echo "Starting Jupyter notebook server..."
	jupyter notebook notebooks/

lint:
	@echo "Linting code..."
	flake8 src/ --max-line-length=100
	@echo "✓ Linting complete"

format:
	@echo "Formatting code..."
	black src/ notebooks/
	@echo "✓ Formatting complete"

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker-compose build
	@echo "✓ Docker image built"

docker-up:
	@echo "Starting Docker containers..."
	@echo "Jupyter Lab will be available at http://localhost:8888"
	docker-compose up

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "✓ Containers stopped"

docker-test:
	@echo "Running tests in Docker..."
	docker-compose run --rm tests
	@echo "✓ Tests complete"

docker-shell:
	@echo "Opening shell in container..."
	docker-compose run --rm jupyter bash

docker-clean:
	@echo "Cleaning Docker resources..."
	docker-compose down --rmi all -v
	@echo "✓ Docker cleanup complete"

.DEFAULT_GOAL := help
