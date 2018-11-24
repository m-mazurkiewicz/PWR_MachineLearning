import os
import sys
from shutil import copy2

# copy2('test/test.py', 'test1/test1.py')

test = [x for x in os.walk('NN_summary')]

for network in test[0][1]:
    copy2('report_template.tex','NN_summary/'+network)
    copy2('pythontex/pythontex_engines.py','NN_summary/'+network)
    copy2('pythontex/pythontex_utils.py','NN_summary/'+network)
    copy2('pythontex/pythontex3.py','NN_summary/'+network)