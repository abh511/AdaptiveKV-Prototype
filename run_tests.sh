#!/bin/bash
#
# AdaptiveKV Test Suite Runner
# Runs all validation tests to verify the implementation
#

set -e  # Exit on first error

echo "════════════════════════════════════════════════════════════════"
echo "  AdaptiveKV Test Suite"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if python -m pytest tests/ -v; then
    echo -e "${GREEN}✓ Test suite PASSED${NC}"
else
    echo -e "${RED}✗ Test suite FAILED${NC}"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Review test outputs in outputs/ directory"
echo "  2. Run dashboard: ./run_dashboard.sh"
echo "  3. Check FINAL_VALIDATION_REPORT.md for details"
echo ""
