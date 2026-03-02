#!/usr/bin/env python3
"""Run figure generation scripts and log output."""
import sys
import os

# Redirect output to log file
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_log.txt')
log_file = open(log_path, 'w', encoding='utf-8')
sys.stdout = log_file
sys.stderr = log_file

try:
    print("=" * 60)
    print("Running generate_lifecycle_pdf.py ...")
    print("=" * 60)
    
    # Run lifecycle figure
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import generate_lifecycle_pdf
    generate_lifecycle_pdf.main()
    
    print("\n" + "=" * 60)
    print("Running fig_paper_final.py ...")
    print("=" * 60)
    
    # Run fig_paper_final
    fig_final_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 
        'scripts', 'xibeiyuzhicai', 'output',
        'xibeiyuzhicai_baseline_20260223_145442_14B效果不错',
        '现象验证-最终'
    )
    fig_final_dir = os.path.normpath(fig_final_dir)
    sys.path.insert(0, fig_final_dir)
    import fig_paper_final
    fig_paper_final.main()
    
    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    log_file.close()
