# =============================================================================
# Refund Sentinel — Makefile
# =============================================================================
# Targets marked [FUTURE] depend on files not yet implemented.
# They are listed here for planning purposes and will fail if run now.
# =============================================================================

.DEFAULT_GOAL := help

# Detect the Python executable (prefer python3)
PYTHON := python3
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
	@echo "    test           Run the test suite [FUTURE — no tests yet]"
	@echo "    test-unit      Run unit tests only [FUTURE]"
	@echo "    test-int       Run integration tests only [FUTURE]"
	@echo ""
	@echo "  Data and training [FUTURE]"
	@echo "    seed-demo      Generate and load the demo dataset"
	@echo "    gen-datasets   Generate training and evaluation datasets"
	@echo "    train          Train the ML model"
	@echo "    evaluate       Run the full evaluation suite"
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
	@echo "[FUTURE] No tests exist yet. This target will run pytest once Phase 2 is implemented."
	@echo "Run after Phase 2: pytest backend/tests/"

.PHONY: test-unit
test-unit:
	@echo "[FUTURE] pytest backend/tests/unit/"

.PHONY: test-int
test-int:
	@echo "[FUTURE] pytest backend/tests/integration/"

# =============================================================================
# DATA AND TRAINING (FUTURE)
# =============================================================================

.PHONY: seed-demo
seed-demo:
	$(PYTHON) -m backend.app.simulator.cli

.PHONY: gen-datasets
gen-datasets:
	@echo "[FUTURE] $(PYTHON) scripts/generate_datasets.py"
	@echo "Implement after Phase 8 (full simulator)."

.PHONY: train
train:
	@echo "[FUTURE] $(PYTHON) scripts/train_model.py"
	@echo "Implement after Phase 9 (ML model)."

.PHONY: evaluate
evaluate:
	@echo "[FUTURE] $(PYTHON) scripts/run_evaluation.py"
	@echo "Implement after Phase 10 (evaluation framework)."

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