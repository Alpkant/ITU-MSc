*One can test the Question2 by choosing either of the options:
 1-) Any .mat adjacency matrix can be given by running "python Question2.py --path your_.mat_file_relative_path"
 2-) By running "python Question2.py --random True" random matrix will be initialized and tested.
 More preferred way to test is by giving the path of .mat file.
 Also there is an option the test the weight calculation of the centrality measure.
 Weight calculation methods explained in the project report.
 1-) Executing python Question2.py --path --path your_.mat_file_relative_path" --weight 1
 will calculate the centralities with constant weights.
 2-) Executing python Question2.py --path --path your_.mat_file_relative_path" --weight 2
 will calculate the centralities with frequency based weights. 
 3-) Default weight argument is 2 

 Note that you can also use these weight argument with random graphs too. So random and path arguments selects the graph creation procedure,
 weight argument select the weight calculation procedure.


Requirements:
- matplotlib.pyplot 
- snap 
- scipy.io 
- numpy 
- argparse