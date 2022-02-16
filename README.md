# MF-NBEA
Implementation of Multi-Level Data Fusion and Neural Basis Expansion Analysis for interpretable (MF-NBEA) solar power time series forecasting

MF-NBEA is a an extension of the successful N-BEATS model to a solar power forecasting problem, which is a multivariate context with temporal and spatial characterstics. 

## Data
The raw datasets are publicly available online: http://www.nemweb.com.au/REPORTS/ARCHIVE/Dispatch\_SCADA/
the weather information from : https://solcast.com/

## Installation


To run the code: 
1. Edit  create_config_MF-NBEA.py with required configurations 
2. Run create_config_MF-NBEA.py  , this will generate jason files of all required configurations to use. All jason files will be saved in \model_configs.
3. After creating all the configurations you are ready to run the code.. just run the following command:

```bash
  python run.py 
```

To provide correct data files names, modify  \data_loader\Data_util.py 


