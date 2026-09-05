# =============================================================================
# Refund Sentinel — Makefile
# =============================================================================

.DEFAULT_GOAL := help

# Detect the Python executable (prefer python3)
PYTHON := python
PIP    := pip

# Backend source root
BACKEND := backend

# =============================================================================
# HELP
# =============================================================================

.PHONY: help
help:
	@echo ""
	@echo "Refund Sentinel — available make targets"
	@echo ""
	@echo "  Development environment"
	@echo "    db-up          Start PostgreSQL (Docker)"
	@echo "    db-down        Stop PostgreSQL"
	@echo "    db-reset       Stop, remove volume, and restart PostgreSQL"
	@echo "    db-logs        Tail PostgreSQL logs"
	@echo "    db-shell       Open a psql shell inside the running container"
	@echo ""
	@echo "  Python environment"
	@echo "    install        Install backend dependencies into the active virtualenv"
	@echo ""
	@echo "  Testing"
	@echo "    test           Run full test suite"
	@echo "    test-unit      Run unit tests only"
	@echo "    test-int       Run integration tests only"
	@echo ""
	@echo "  Data and Training"
	@echo "    seed-demo      Populate database with 5 canonical demo scenarios"
	@echo "    gen-datasets   Generate training, validation, and held-out test datasets"
	@echo "    train          Train the supervised ML model"
	@echo "    evaluate       Run the comparative 3-baseline evaluation suite"
	@echo ""
	@echo "  Cleanup"
	@echo "    clean          Remove Python cache files"
	@echo ""

# =============================================================================
# DATABASE
# =============================================================================

.PHONY: db-up
db-up:
	docker compose up -d db
	@echo "Waiting for PostgreSQL to be healthy..."
	@until docker compose exec db pg_isready -U sentinel -d refund_sentinel > /dev/null 2>&1; do \
		sleep 1; \
	done
	@echo "PostgreSQL is ready."

.PHONY: db-down
db-down:
	docker compose stop db

.PHONY: db-reset
db-reset:
	docker compose down -v
	docker compose up -d db
	@echo "Waiting for PostgreSQL to be healthy..."
	@until docker compose exec db pg_isready -U sentinel -d refund_sentinel > /dev/null 2>&1; do \
		sleep 1; \
	done
	@echo "PostgreSQL restarted with a clean volume."

.PHONY: db-logs
db-logs:
	docker compose logs -f db

.PHONY: db-shell
db-shell:
	docker compose exec db psql -U sentinel -d refund_sentinel

# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================

.PHONY: install
install:
	$(PIP) install -r $(BACKEND)/requirements.txt

# =============================================================================
# TESTING
# =============================================================================

.PHONY: test
test:
	pytest $(BACKEND)/tests/ -v

.PHONY: test-unit
test-unit:
	pytest $(BACKEND)/tests/unit/ -v

.PHONY: test-int
test-int:
	pytest $(BACKEND)/tests/integration/ -v

# =============================================================================
# DATA AND TRAINING
# =============================================================================

.PHONY: seed-demo
seed-demo:
	$(PYTHON) scripts/seed_demo.py

.PHONY: gen-datasets
gen-datasets:
	$(PYTHON) scripts/generate_datasets.py

.PHONY: train
train:
	$(PYTHON) scripts/train_model.py

.PHONY: evaluate
evaluate:
	$(PYTHON) scripts/run_evaluation.py

# =============================================================================
# CLEANUP
# =============================================================================

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Python cache files removed."