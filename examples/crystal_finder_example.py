"""
This example shows how to run the crystal finder algorithm which finds crystals from a 2D array
containing number of spots. Additionally the CrystalFinder3D class is able to infer the volume of
a crustal based on 2 grid scans (usually the flat and edge grid scans)
"""

from mx3_beamline_library.science.optical_and_loop_centering.crystal_finder import CrystalFinder, CrystalFinder3D
import numpy as np
import matplotlib.pyplot as plt

# Simulate number-of-spots data
x, y = np.meshgrid(np.linspace(-1,1,30), np.linspace(-1,1,30))
d = np.sqrt(x*x+y*y)
sigma, mu = 0.2, 0.0
number_of_spots = 100*np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

# Run the crystal finder for the first grid scan
crystal_finder = CrystalFinder(number_of_spots=number_of_spots, threshold=5)
coords_flat, distance_flat, max_number_of_spots_flat = crystal_finder.plot_crystal_finder_results(save=True, filename="flat")

# Run the crystal finder for the second grid scan
crystal_finder = CrystalFinder(number_of_spots=number_of_spots, threshold=5)
coords_edge, distance_edge, max_number_of_spots_edge = crystal_finder.plot_crystal_finder_results(save=True, filename="edge")

# Find the crystals volumes
crystal_finder_3d = CrystalFinder3D(crystal_coordinates_flat=coords_flat, crystal_coordinates_edge=coords_edge, dist_flat=distance_flat, dist_edge=distance_edge)
crystal_volumes = crystal_finder_3d.plot_crystals(save=True,filename="3d_results")