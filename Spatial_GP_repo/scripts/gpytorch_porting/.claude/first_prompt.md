
General focus of the first porting attempt to gpytorch of the spatial_gp_repo

Read all the claude.md files you find in /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo .  we are working with a gaussian process algorithm to predict neural responses to natural stimulations ( images ). I have worked on the code for years and despite not being enormously complicated the codebase has become too big and messy.  

Everything revolves around function GP_utils.varGP() for the fitting. It is a very long and complicated function so do not get lost in the details of it. 

The latex document you should focus on is ~/IDV_code/Papers/latex_summaries/gaussian_process_theory.tex . it is very pedagocical but it explains exactly the gp model we are using. 

Details of the used kernel are in /home/idv-eqs8-pza/IDV_code/Papers/latex_summaries/ ( dont read the derivatives section )  the aim of this session is not to provide code at all, its about exploring the math and the codebase to find a way to port this code into gpytorch. 

There are several things to consider about the codebase:

1) gpytorch does not provide the poisson likelihood out of the box, and the same goes for the arcosine kernel. Consider if its a good idea to start from a linear kernel and focus on the codebase structure. we have no rush, the priority is precision 

2) We need a standardised " test " , or even a simple fitting script " that should give the same results with the two implementations, a first draft for  this can be /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/scripts/one_cell_fit.py , consider if this is a good starting point. We will have to run some tests for the intermediate building blocks we will work on, this is just the " final goal " script.

3) Some parts of the code have indeed been already "ported" . For instance , there is a /home/idv-eqs8-pza/IDV_code/ClosedLoopProject/gaussian_processes/Spatial_GP_repo/kernels in which there might be some useful code. read it carefully and tell me what is useful and what is not.

Here are the things to consider for your working session:

1) The focus here is to take the perspective of a sofware developer and a scientist too. We are coding all this for science, so there is no need to make the codebase explode with unittests , super complicated classes etc. I am not a professional developer and I need to be able to comprehend what we do, and put hands on the code myself if needed. consider this when proposing ideas

2) Despite this, Do not accept everything I write as ground truth, if you think something I propose is not the optimal way, let me know , just keep in mind to not overcomplicate things. WE ARE AIMING FOR A FIRST VERSION IMPLEMENTATION OF THIS. we will have time to add functionalities later if needed.

3) Your working goal at the beginning is to define how to takle the problem with claude code. Should i tell you to devise a plan on how to explore the codebase? it is pretty big and I dont know what is the best practice in this case for claude code. ultrahink very deep about the full project


















