import numpy as np

swissci_lowprofile ={
    "type":"swissci_lowprofile",
    "origin_uv":(0, 0),
    "dx":9,
    "dy":8.95,
    "wellx":3.6,
    "welly":4.25,
    "subgrid_rows":2,
    "subgrid_cols":2,
    "reference_offset": {"x":0,"y":1.7,"z":-1.5},
    "calibration_points" :  {'A1': np.array([30.19899,4.798,2.095241]), 'H1': np.array([30.1001,67.5,2.114361]), 'A12': np.array([-68.49988,4.798,1.750001])},
    "reference_points": {'A1': np.array([30.99032 ,  3.198   , -0.499991]), 'H1': np.array([30.80034 , 65.848   , -0.499981]), 'A12': np.array([-68.5127  ,   3.693   ,  -0.499991])},
    "scan":{"start":1, "stop":3},
    "depth": -1.5,
}

swissci_highprofile ={
    "type":"swissci_highprofile",
    "origin_uv":(0, 0),
    "dx":9,
    "dy":8.95,
    "wellx":3.6,
    "welly":4.25,
    "subgrid_rows":2,
    "subgrid_cols":2,
    "reference_offset": {"x":-3,"y":1.4,"z":-1.5},
    "calibration_points" : {'A1': np.array([27.50007,2.8,2.300075]), 'H1': np.array([27.80008,63.5,2.300035]), 'A12': np.array([-71.49997,2.4,2.300055])},
    "reference_points" : {'A1': np.array([27.50007,2.8,2.300075]), 'H1': np.array([27.80008,63.5,2.300035]), 'A12': np.array([-71.49997,2.4,2.300055])},
    "scan":{"start":1.5, "stop":3},
    "depth": -1.5,
}

mitegen_insitu ={
    "type":"mitegen_insitu",
    "origin_uv":(0, 0),
    "dx":9.1,
    "dy":8.9,
    "wellx":3.4,
    "welly":1.6,
    "subgrid_rows":2,
    "subgrid_cols":2,
    "reference_offset": {"x":-5,"y":3,"z":1.8},
    "points": ["A1", "H1", "H12"],
    "calibration_points" : {'A1': np.array([28.0002,0.3,-1.399945]), 'H1': np.array([27.88719,62.5,-1.437715]), 'H12': np.array([-71,61.6,-1.25])},
    "reference_points" : {'A1': np.array([28.0002,0.3,-1.399945]), 'H1': np.array([27.88719,62.5,-1.237]), 'H12': np.array([-72.5,63.5,-1.75])},
    "scan":{"start":-2, "stop":-0.5},
    "depth": -0.5,
}

mrc ={
    "type":"mrc",
    "origin_uv":(0, 0),
    "dx":9.1,
    "dy":8.9,
    "wellx":4.5,
    "welly":3.8,
    "subgrid_rows":2,
    "subgrid_cols":2,
    "reference_offset": {"x":0,"y":1.7,"z":-1.5},
    "calibration_points" :  {'A1': np.array([30.19899,4.798,1.9]), 'H1': np.array([30.1001,67.5,1.852]), 'A12': np.array([-68.49988,4.798,1.8])},
    "reference_points": {'A1': np.array([30.99032 ,  3.198   , 1.9]), 'H1': np.array([30.80034 , 65.848   , 1.852]), 'A12': np.array([-68.5127  ,   3.693   ,  1.8])},
    "scan":{"start":1, "stop":3},
    "depth": 0,
}
