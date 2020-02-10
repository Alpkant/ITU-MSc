*To test Question 1.1, 1.2, 1.3 there are two options:
 1-) Any .mat adjacency matrix can be given by running "python Question1_except4.py --path your_.mat_file_relative_path"
 2-) By running "python Question1_except4.py --random True" random matrix will be initialized and tested.
 More preferred way to test is by giving the path of .mat file.

*To test the Question 1.4 same two options are available. 
 1-) "python Question1_4.py --path your_.mat_file_relative_path"
 2-) "python Question1_4.py --random True"
 Test script will print the percolation threshold to the terminal. Also the plot of the connected components will be shown and saved into same folder.

Requirements:
- matplotlib.pyplot 
- snap
- scipy.io
- numpy 
- argparse