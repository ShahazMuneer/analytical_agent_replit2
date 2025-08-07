#!/bin/bash

# Enterprise Analytics Hub Startup Script
# Production-ready startup with health checks and logging

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Configuration
APP_NAME="Enterprise Analytics Hub"
APP_VERSION="2.0.0"
PYTHON_VERSION="3.11"
REQUIRED_MEMORY_MB=512
MAX_STARTUP_TIME=120

# Environment variables with defaults
export STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}
export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-5000}
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}
export PYTHONPATH=${PYTHONPATH:-/app}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# Startup banner
echo "
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║               🚀 Enterprise Analytics Hub 🚀                     ║
║                                                                  ║
║                          Version ${APP_VERSION}                           ║
║               AI-Powered Business Intelligence Platform          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"

log "Starting ${APP_NAME} v${APP_VERSION}..."

# Pre-flight checks
perform_preflight_checks() {
    log "Performing pre-flight checks..."
    
    # Check Python version
    CURRENT_PYTHON=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$CURRENT_PYTHON" != "$PYTHON_VERSION" ]]; then
        warning "Python version mismatch. Expected: ${PYTHON_VERSION}, Current: ${CURRENT_PYTHON}"
    else
        success "Python version check passed (${CURRENT_PYTHON})"
    fi
    
    # Check available memory
    if command -v free > /dev/null; then
        AVAILABLE_MEMORY=$(free -m | awk 'NR==2{print $7}')
        if [[ $AVAILABLE_MEMORY -lt $REQUIRED_MEMORY_MB ]]; then
            warning "Low memory available: ${AVAILABLE_MEMORY}MB (recommended: ${REQUIRED_MEMORY_MB}MB+)"
        else
            success "Memory check passed (${AVAILABLE_MEMORY}MB available)"
        fi
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df /app | awk 'NR==2 {print $4}')
    if [[ $AVAILABLE_SPACE -lt 1048576 ]]; then  # 1GB in KB
        warning "Low disk space available"
    else
        success "Disk space check passed"
    fi
    
    # Check required environment variables
    if [[ -z "$GROQ_API_KEY" ]] || [[ "$GROQ_API_KEY" == "gsk_default_key" ]]; then
        warning "GROQ_API_KEY not set or using default value"
        warning "Some AI features may not work properly"
    else
        success "GROQ_API_KEY is configured"
    fi
    
    # Check network connectivity (if external APIs are used)
    if command -v curl > /dev/null; then
        if curl -s --max-time 5 https://api.groq.com > /dev/null; then
            success "Network connectivity check passed"
        else
            warning "Cannot reach external APIs - check network connectivity"
        fi
    fi
}

# Initialize application directories and files
initialize_app() {
    log "Initializing application..."
    
    # Create necessary directories
    mkdir -p /app/data /app/logs /app/temp
    
    # Set proper permissions
    chmod 755 /app/data /app/logs /app/temp
    
    # Initialize log file
    touch /app/logs/analytics_hub.log
    
    # Check if database exists, if not initialize it
    if [[ ! -f "/app/analytics.db" ]] && [[ ! -f "/app/data/analytics.db" ]]; then
        log "Database not found, initializing..."
        python /app/init_db.py
        if [[ $? -eq 0 ]]; then
            success "Database initialized successfully"
        else
            error "Database initialization failed"
            exit 1
        fi
    else
        success "Database found"
    fi
    
    # Validate database integrity
    log "Validating database integrity..."
    python -c "
import sqlite3
import sys
try:
    conn = sqlite3.connect('/app/analytics.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM sales_product_customer_view')
    count = cursor.fetchone()[0]
    print(f'Database validation passed: {count} records found')
    conn.close()
except Exception as e:
    print(f'Database validation failed: {e}')
    sys.exit(1)
"
    if [[ $? -eq 0 ]]; then
        success "Database validation passed"
    else
        error "Database validation failed"
        exit 1
    fi
}

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    log "Performing health check..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:${STREAMLIT_SERVER_PORT}/_stcore/health > /dev/null 2>&1; then
            success "Health check passed on attempt ${attempt}"
            return 0
        fi
        
        log "Health check attempt ${attempt}/${max_attempts} failed, retrying in 2 seconds..."
        sleep 2
        ((attempt++))
    done
    
    error "Health check failed after ${max_attempts} attempts"
    return 1
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    
    # Kill any background processes
    jobs -p | xargs -r kill
    
    # Clean temporary files
    rm -rf /app/temp/*
    
    log "Cleanup completed"
}

# Signal handlers
handle_sigterm() {
    log "Received SIGTERM signal, initiating graceful shutdown..."
    cleanup
    exit 0
}

handle_sigint() {
    log "Received SIGINT signal, initiating graceful shutdown..."
    cleanup
    exit 0
}

# Set up signal handlers
trap handle_sigterm SIGTERM
trap handle_sigint SIGINT

# Main execution
main() {
    # Perform pre-flight checks
    perform_preflight_checks
    
    # Initialize application
    initialize_app
    
    # Start the application
    log "Starting Streamlit application..."
    log "Server will be available at: http://0.0.0.0:${STREAMLIT_SERVER_PORT}"
    
    # Run Streamlit in background to allow health checks
    streamlit run /app/app.py \
        --server.headless=${STREAMLIT_SERVER_HEADLESS} \
        --server.address=${STREAMLIT_SERVER_ADDRESS} \
        --server.port=${STREAMLIT_SERVER_PORT} \
        --browser.gatherUsageStats=${STREAMLIT_BROWSER_GATHER_USAGE_STATS} \
        --logger.level=${LOG_LEVEL} &
    
    STREAMLIT_PID=$!
    log "Streamlit started with PID: ${STREAMLIT_PID}"
    
    # Wait for application to be ready
    sleep 10
    
    # Perform health check
    if health_check; then
        success "${APP_NAME} is running successfully!"
        success "🌐 Application URL: http://localhost:${STREAMLIT_SERVER_PORT}"
        success "📊 Ready to serve analytics requests"
    else
        error "Application failed to start properly"
        kill $STREAMLIT_PID 2>/dev/null || true
        exit 1
    fi
    
    # Monitor the application
    log "Monitoring application health..."
    while kill -0 $STREAMLIT_PID 2>/dev/null; do
        # Periodic health check (every 5 minutes)
        sleep 300
        if ! health_check; then
            warning "Health check failed during runtime"
        fi
    done
    
    warning "Streamlit process has stopped"
    exit 1
}

# Error handling
set -e
set -o pipefail

# Execute main function
main "$@"
