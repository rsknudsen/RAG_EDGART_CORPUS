# EDGAR Crawler & RAG Solution

## Author
Rasmus Knudsen

## Overview
-Data in aig-filings
-Code in .py files (run rag_solution.py to get results)
  -Run data_ingest.py for new data (this step can be skipped as data is already downloaded to folder)
  -Run rag_solution.py to generate results
  -Run evaluate_results.py to get merged_results.csv, a table comparing results to ground truth. 
-Ground truth and results in json files. Ground truth was generated using ChatGPT. 
-merged_results.csv for evaluation (human)
-(NOT INCLUDED) store your OpenAI environment key. 

## Results
2/5 years correct for total employees, not retrieved for recent years. 
Total assets retrieves the wrong metric, needs tuning. 
