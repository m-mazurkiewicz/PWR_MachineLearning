import os
import sys
from shutil import copy2

# copy2('test/test.py', 'test1/test1.py')

test = [x for x in os.walk('NN_summary')]

# for nn in test[0][1]:
#     print(nn)

for network in test[0][1]:
    os.chdir("NN_summary/"+network)
    # print(network)
    # print("\"NN_summary/"+network+"\"")
    # print("pdflatex -output-directory \"NN_summary/"+network+"\" \"NN_summary/"+network+'/'+'report_template'+'.tex\"')
    # print("python pythontex/pythontex.py \"NN_summary/"+network+'/'+'report_template'+'.tex\"')
    # print("pdflatex -output-directory \"NN_summary/"+network+"\" \"NN_summary/"+network+'/'+'report_template'+'.tex\"')
    os.system("pdflatex report_template.tex")
    os.system("python pythontex3.py report_template.tex")
    os.system("pdflatex report_template.tex")
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))
