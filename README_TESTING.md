# Comprehensive Testing Notebook for Propeller and Powering Models

This directory contains comprehensive testing tools for the standalone propeller and powering models.

## Files

- `test_models_comprehensive.py` - Comprehensive testing script with all test sections
- `test_models_comprehensive.ipynb` - Jupyter notebook version (to be created)

## Usage

### Option 1: Run as Python Script
```bash
python test_models_comprehensive.py
```

### Option 2: Convert to Jupyter Notebook
1. Open Jupyter Notebook or JupyterLab
2. Create a new notebook
3. Copy sections from `test_models_comprehensive.py` into cells
4. Or use: `jupyter nbconvert --to notebook --execute test_models_comprehensive.py`

### Option 3: Use in Jupyter
The script is structured with clear section markers (comments starting with `# =====`). You can:
- Copy each section into a separate notebook cell
- Add markdown cells between sections for documentation
- Run cells individually for interactive testing

## Test Sections Included

1. **Setup** - Physics constants and configuration
2. **Loadcases** - Test scenarios with different speeds
3. **Thrust Vectors** - Thrust requirements (including negative/reverse)
4. **Propeller Configurations** - Multiple propeller models
5. **Powering Configurations** - DD, DE, PTI/PTO modes
6. **Basic Tests** - Thrust from power
7. **RPM-Based Tests** - Power/thrust from RPM
8. **Inverse Tests** - RPM from thrust/power
9. **Efficiency Analysis** - J coefficient and efficiency curves
10. **Performance Curves** - KT, KQ, efficiency plots
11. **Edge Cases** - Zero speed, negative thrust, validation
12. **Full Integration** - Complete power flow testing
13. **EST Power Impact** - Electrical storage technology testing
14. **Multiple Propeller** - Single vs double propeller comparison
15. **PTI/PTO Detailed** - Mode transition analysis
16. **Optimization** - Optimal RPM and power scenarios
17. **Visualizations** - 12 comprehensive plots
18. **Data Export** - CSV export of all results
19. **Summary Statistics** - Test statistics and validation

## Output Files

The script generates:
- `test_results_comprehensive.png` - Comprehensive visualization plots
- `results_*.csv` - CSV files with detailed test results for each section

## Key Features Tested

### Propeller Models
- ✅ Wageningen B-series
- ✅ Custom curves
- ✅ Simple thrust model
- ✅ Variable efficiency (speed-dependent)
- ✅ Single vs multiple propellers

### Powering Modes
- ✅ Diesel-Direct (DD)
- ✅ Diesel-Electric (DE)
- ✅ PTI/PTO

### Analysis Types
- ✅ Efficiency vs J curves
- ✅ Performance curves (KT, KQ)
- ✅ Fuel consumption analysis
- ✅ Power flow validation
- ✅ Edge case handling
- ✅ Optimization scenarios

## Notes

- All tests include error handling
- Results are exported to CSV for further analysis
- Comprehensive visualizations are generated
- Edge cases and validation are included
- Multiple propeller and powering configurations are tested

