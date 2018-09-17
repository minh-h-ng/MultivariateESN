#!/bin/bash
# Virtual Environment
source ~/keras-tf-venv3/bin/activate

cd /home/minh/Desktop/vb_linear
/usr/local/MATLAB/R2017b/bin/matlab -nodesktop -nosplash -r 'vb_examples;quit;'
cd /home/minh/PycharmProjects/MultivariateESN